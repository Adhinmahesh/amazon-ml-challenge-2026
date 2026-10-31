#!/usr/bin/env python3
"""
🏢 Amazon ML Challenge 2026 — Business Entity Resolution
═══════════════════════════════════════════════════════════
Memory-Optimized Pipeline for 15 GB RAM Local Machines

Strategy: Disk-based country partitioning
  Phase 1 — Split all source TSVs by country onto disk (stream, never all in RAM)
  Phase 2 — Load ONE country at a time: block → extract features → free
  Phase 3 — Train LightGBM (features are tiny relative to raw data)
  Phase 4 — Country-by-country test inference
  Phase 5 — Save output files & validate

Run with:
    /home/user/Desktop/haws/venv/bin/python /home/user/Desktop/haws/solution_local.py
"""

# ================================================================
# IMPORTS
# ================================================================
import os, gc, re, sys, time, warnings, math, glob, shutil, json, pickle
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sparse_dot_topn import awesome_cossim_topn
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein as Lev
from rapidfuzz.distance import JaroWinkler as JW

import lightgbm as lgb
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("✅ All imports successful")

# ================================================================
# CONFIGURATION
# ================================================================
class Config:
    """Central configuration — edit paths & hyper-params here."""

    # ── Paths ────────────────────────────────────────────────
    BASE_DIR       = "/home/user/Desktop/haws/6ab10eb3b23ba_student_resource/student_resource"
    TRAIN_DIR      = os.path.join(BASE_DIR, "dataset", "train")
    TEST_DIR       = os.path.join(BASE_DIR, "dataset", "test")
    OUTPUT_DIR     = os.path.join(BASE_DIR, "output")
    PARTITION_DIR  = os.path.join(BASE_DIR, "partitions")

    # ── Blocking ─────────────────────────────────────────────
    TOP_K              = 25
    TFIDF_MIN_SIM      = 0.03
    TFIDF_MAX_FEATURES = 50_000
    TFIDF_NGRAM_RANGE  = (3, 4)
    TFIDF_MIN_DF       = 3
    BLOCK_CHUNK_SIZE   = 1_500

    # ── Sampling (training / tuning) ─────────────────────────
    TRAIN_SAMPLE_FRAC  = 0.015   # fraction of S1 entities to sample per country
    NEG_POS_RATIO      = 3

    # ── Optuna ───────────────────────────────────────────────
    OPTUNA_TRIALS      = 25
    OPTUNA_TIMEOUT     = 1800

    # ── Final model ──────────────────────────────────────────
    FINAL_N_ESTIMATORS = 2000
    EARLY_STOP_ROUNDS  = 100
    THRESHOLD_GRID     = np.arange(0.10, 0.96, 0.02)

    # ── Misc ─────────────────────────────────────────────────
    RANDOM_SEED        = 42
    N_JOBS             = -1
    READ_CHUNK_SIZE    = 300_000   # rows per chunk when streaming TSVs

CFG = Config()
os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
np.random.seed(CFG.RANDOM_SEED)

print("✅ Configuration loaded")

# ================================================================
# TEXT NORMALISATION HELPERS
# ================================================================

_NAME_SUBS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bcorp\b"),       "corporation"),
    (re.compile(r"\binc\b"),        "incorporated"),
    (re.compile(r"\bltd\b"),        "limited"),
    (re.compile(r"\bpvt\b"),        "private"),
    (re.compile(r"\bllc\b"),        "limited liability company"),
    (re.compile(r"\bllp\b"),        "limited liability partnership"),
    (re.compile(r"\bco\b"),         "company"),
    (re.compile(r"\bassoc\b"),      "association"),
    (re.compile(r"\bintl\b"),       "international"),
    (re.compile(r"\bnatl\b"),       "national"),
    (re.compile(r"\bmfg\b"),        "manufacturing"),
    (re.compile(r"\bsvc\b"),        "service"),
    (re.compile(r"\bsvcs\b"),       "services"),
    (re.compile(r"\bgvt\b"),        "government"),
    (re.compile(r"\bgovt\b"),       "government"),
    (re.compile(r"\buniv\b"),       "university"),
    (re.compile(r"\bdba\b"),        ""),
    (re.compile(r"\bt/a\b"),        ""),
]

_ADDR_SUBS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bst\b"),         "street"),
    (re.compile(r"\brd\b"),         "road"),
    (re.compile(r"\bave\b"),        "avenue"),
    (re.compile(r"\bblvd\b"),       "boulevard"),
    (re.compile(r"\bdr\b"),         "drive"),
    (re.compile(r"\bln\b"),         "lane"),
    (re.compile(r"\bct\b"),         "court"),
    (re.compile(r"\bpl\b"),         "place"),
    (re.compile(r"\bpkwy\b"),       "parkway"),
    (re.compile(r"\bhwy\b"),        "highway"),
    (re.compile(r"\bcir\b"),        "circle"),
    (re.compile(r"\bter\b"),        "terrace"),
    (re.compile(r"\btrce\b"),       "trace"),
    (re.compile(r"\bsq\b"),         "square"),
    (re.compile(r"\bapt\b"),        "apartment"),
    (re.compile(r"\bste\b"),        "suite"),
    (re.compile(r"\bfl\b"),         "floor"),
    (re.compile(r"\bbldg\b"),       "building"),
    (re.compile(r"\bn\b"),          "north"),
    (re.compile(r"\bs\b"),          "south"),
    (re.compile(r"\be\b"),          "east"),
    (re.compile(r"\bw\b"),          "west"),
    (re.compile(r"\bne\b"),         "northeast"),
    (re.compile(r"\bnw\b"),         "northwest"),
    (re.compile(r"\bse\b"),         "southeast"),
    (re.compile(r"\bsw\b"),         "southwest"),
    (re.compile(r"\bpo box\b"),     ""),
    (re.compile(r"\bp\.?o\.?\s*box\b"), ""),
]

_RE_AMP      = re.compile(r"&")
_RE_NON_ALNUM = re.compile(r"[^\w\s]", re.UNICODE)
_RE_MULTI_WS = re.compile(r"\s+")
_RE_LEADING  = re.compile(r"^[^a-zA-Z0-9\u0900-\u097F\u00C0-\u024F]+")


def _base_normalise(text: str) -> str:
    if pd.isna(text) or not str(text).strip():
        return ""
    text = str(text).lower().strip()
    text = _RE_LEADING.sub("", text)
    text = _RE_AMP.sub(" and ", text)
    text = _RE_NON_ALNUM.sub(" ", text)
    text = _RE_MULTI_WS.sub(" ", text).strip()
    return text


def normalise_name(text: str) -> str:
    text = _base_normalise(text)
    if not text:
        return ""
    for pat, repl in _NAME_SUBS:
        text = pat.sub(repl, text)
    text = _RE_MULTI_WS.sub(" ", text).strip()
    return text


def normalise_address(text: str) -> str:
    text = _base_normalise(text)
    if not text:
        return ""
    for pat, repl in _ADDR_SUBS:
        text = pat.sub(repl, text)
    text = _RE_MULTI_WS.sub(" ", text).strip()
    return text


print("✅ Normalisation functions defined")

# ================================================================
# PHASE 1: PARTITION DATA BY COUNTRY TO DISK
# ================================================================

def _safe_country(country: str) -> str:
    """Filesystem-safe country name."""
    return re.sub(r"[^\w]", "_", str(country).strip())


def partition_source_file(filepath: str, out_dir: str, prefix: str,
                          add_source_col: Optional[str] = None) -> int:
    """
    Stream-read a source TSV, normalise text, write country-specific files.
    Never holds more than one chunk (~300K rows) in memory.
    """
    written: Set[str] = set()
    total = 0

    reader = pd.read_csv(filepath, sep="\t", chunksize=CFG.READ_CHUNK_SIZE,
                         dtype=str, na_filter=False)

    for chunk in tqdm(reader, desc=f"    {os.path.basename(filepath)}", leave=False):
        # Normalise in-place on the chunk
        chunk["name_norm"]  = chunk["business_name"].apply(normalise_name)
        chunk["addr_norm"]  = chunk["business_address"].apply(normalise_address)
        chunk["combined"]   = chunk["name_norm"] + " " + chunk["addr_norm"]

        if add_source_col:
            chunk["source"] = add_source_col

        for country, grp in chunk.groupby("country"):
            safe = _safe_country(country)
            fpath = os.path.join(out_dir, f"{prefix}_{safe}.tsv")
            header = fpath not in written
            grp.to_csv(fpath, sep="\t", index=False, header=header, mode="a")
            written.add(fpath)

        total += len(chunk)
        del chunk; gc.collect()

    return total


def run_partitioning():
    """Split all source files by country into separate TSVs on disk."""
    train_part = os.path.join(CFG.PARTITION_DIR, "train")
    test_part  = os.path.join(CFG.PARTITION_DIR, "test")

    # Wipe old partitions
    for d in [train_part, test_part]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    print("\n" + "=" * 60)
    print("PHASE 1: Partitioning data by country to disk")
    print("=" * 60)
    t0 = time.time()

    # ── Training data ────────────────────────────────────────
    print("  📂 Training Source 1 (S1):")
    n = partition_source_file(
        os.path.join(CFG.TRAIN_DIR, "train_source1.tsv"), train_part, "s1")
    print(f"     → {n:,} rows partitioned")

    print("  📂 Training Source 2 (S23):")
    n = partition_source_file(
        os.path.join(CFG.TRAIN_DIR, "train_source2.tsv"), train_part, "s23",
        add_source_col="source2")
    print(f"     → {n:,} rows partitioned")

    print("  📂 Training Source 3 (S23):")
    n = partition_source_file(
        os.path.join(CFG.TRAIN_DIR, "train_source3.tsv"), train_part, "s23",
        add_source_col="source3")
    print(f"     → {n:,} rows partitioned")

    # ── Test data ────────────────────────────────────────────
    print("  📂 Test Source 1 (S1):")
    n = partition_source_file(
        os.path.join(CFG.TEST_DIR, "test_source1.tsv"), test_part, "s1")
    print(f"     → {n:,} rows partitioned")

    print("  📂 Test Source 2 (S23):")
    n = partition_source_file(
        os.path.join(CFG.TEST_DIR, "test_source2.tsv"), test_part, "s23",
        add_source_col="source2")
    print(f"     → {n:,} rows partitioned")

    print("  📂 Test Source 3 (S23):")
    n = partition_source_file(
        os.path.join(CFG.TEST_DIR, "test_source3.tsv"), test_part, "s23",
        add_source_col="source3")
    print(f"     → {n:,} rows partitioned")

    elapsed = (time.time() - t0) / 60
    print(f"\n  ⏱ Partitioning completed in {elapsed:.1f} min")

    countries = set()
    for f in glob.glob(os.path.join(train_part, "s1_*.tsv")):
        c = os.path.basename(f).replace("s1_", "").replace(".tsv", "")
        countries.add(c)
    print(f"  🌐 Countries found: {sorted(countries)}")
    return sorted(countries)


def get_countries(split: str = "train") -> List[str]:
    """Get the list of countries available for a split."""
    part_dir = os.path.join(CFG.PARTITION_DIR, split)
    countries = set()
    for f in glob.glob(os.path.join(part_dir, "s1_*.tsv")):
        c = os.path.basename(f).replace("s1_", "").replace(".tsv", "")
        countries.add(c)
    return sorted(countries)


def load_partition(split: str, prefix: str, country: str) -> Optional[pd.DataFrame]:
    """Load a single country partition from disk."""
    fpath = os.path.join(CFG.PARTITION_DIR, split,
                         f"{prefix}_{_safe_country(country)}.tsv")
    if not os.path.isfile(fpath):
        return None
    return pd.read_csv(fpath, sep="\t", dtype=str, na_filter=False)


# ================================================================
# GROUND TRUTH LOADING (small dict, stays in memory)
# ================================================================

def load_ground_truth() -> Dict[str, Set[str]]:
    """Return {s1_id: set(matched_ids)} from ground truth TSV."""
    gt_path = os.path.join(CFG.TRAIN_DIR, "train_ground_truth.tsv")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, na_filter=False)
    mapping: Dict[str, Set[str]] = {}
    for _, row in gt.iterrows():
        s1 = row["source1_entity_id"]
        ids_str = row["matched_entity_ids"]
        if ids_str.strip():
            mapping[s1] = set(ids_str.split(","))
        else:
            mapping[s1] = set()
    del gt; gc.collect()
    return mapping


# ================================================================
# BLOCKING FUNCTIONS
# ================================================================

# Maximum S23 rows to process in a single TF-IDF batch.
# If a country has more S23 rows than this, we split into sub-batches.
S23_MAX_BATCH = 2_000_000


def _build_tfidf_and_search(s1_texts, s23_sub, s1_ids, s23_ids,
                             top_k, min_sim, chunk_size):
    """
    Build TF-IDF on a sub-batch of S23, search all S1 against it.
    Returns candidates dict for this sub-batch.
    """
    gc.collect()
    vectoriser = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=CFG.TFIDF_NGRAM_RANGE,
        max_features=30_000,   # Keep low for memory safety
        min_df=CFG.TFIDF_MIN_DF,
        sublinear_tf=True,
        dtype=np.float32,
    )
    tfidf_s23 = vectoriser.fit_transform(s23_sub["combined"])
    gc.collect()

    candidates: Dict[str, List[Tuple[str, float]]] = {}
    n_chunks = math.ceil(len(s1_texts) / chunk_size)

    for i in range(n_chunks):
        start = i * chunk_size
        end   = min(start + chunk_size, len(s1_texts))
        s1_batch = s1_texts.iloc[start:end]
        s1_batch_ids = s1_ids[start:end]

        tfidf_s1 = vectoriser.transform(s1_batch)
        sim_matrix = awesome_cossim_topn(
            tfidf_s1, tfidf_s23.T,
            ntop=top_k,
            lower_bound=min_sim,
            use_threads=True,
            n_jobs=4,
        )

        coo = sim_matrix.tocoo()
        for row, col, score in zip(coo.row, coo.col, coo.data):
            s1_id  = s1_batch_ids[row]
            s23_id = s23_ids[col]
            candidates.setdefault(s1_id, []).append((s23_id, float(score)))

    del vectoriser, tfidf_s23; gc.collect()
    return candidates


def run_blocking_country(
    s1_country: pd.DataFrame,
    s23_country: pd.DataFrame,
    top_k: int = CFG.TOP_K,
    min_sim: float = CFG.TFIDF_MIN_SIM,
    chunk_size: int = CFG.BLOCK_CHUNK_SIZE,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Find top-K candidate S23 entities for each S1 entity in one country.
    If S23 is very large (>2M rows), it is processed in sub-batches
    to keep memory under control.
    """
    if len(s1_country) == 0 or len(s23_country) == 0:
        return {}

    s1_texts = s1_country["combined"]
    s1_ids   = s1_country["entity_id"].values

    all_candidates: Dict[str, List[Tuple[str, float]]] = {}

    if len(s23_country) <= S23_MAX_BATCH:
        # Small enough to process in one go
        print(f"    Processing {len(s23_country):,} S23 records in one batch")
        s23_ids = s23_country["entity_id"].values
        all_candidates = _build_tfidf_and_search(
            s1_texts, s23_country, s1_ids, s23_ids,
            top_k, min_sim, chunk_size)
    else:
        # Split S23 into sub-batches
        n_batches = math.ceil(len(s23_country) / S23_MAX_BATCH)
        print(f"    Processing {len(s23_country):,} S23 records in {n_batches} sub-batches")

        for j in tqdm(range(n_batches), desc="    S23 sub-batches", leave=False):
            s23_start = j * S23_MAX_BATCH
            s23_end   = min(s23_start + S23_MAX_BATCH, len(s23_country))
            s23_sub   = s23_country.iloc[s23_start:s23_end].reset_index(drop=True)
            s23_ids   = s23_sub["entity_id"].values

            sub_cands = _build_tfidf_and_search(
                s1_texts, s23_sub, s1_ids, s23_ids,
                top_k, min_sim, chunk_size)

            # Merge into all_candidates
            for s1_id, cands in sub_cands.items():
                all_candidates.setdefault(s1_id, []).extend(cands)

            del s23_sub, sub_cands; gc.collect()

        # Keep only overall top-K per S1 entity (by score, descending)
        for s1_id in all_candidates:
            cands = all_candidates[s1_id]
            if len(cands) > top_k:
                cands.sort(key=lambda x: -x[1])
                all_candidates[s1_id] = cands[:top_k]

    # Ensure every S1 entity has an entry
    for s1_id in s1_ids:
        all_candidates.setdefault(s1_id, [])

    return all_candidates


print("✅ Blocking functions defined")

# ================================================================
# FEATURE ENGINEERING
# ================================================================

def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def _jaccard(set_a: Set, set_b: Set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def _word_set(text: str) -> Set[str]:
    return set(text.split()) if text else set()


def _len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def compute_pair_features(
    s1_row: pd.Series,
    s23_row: pd.Series,
    tfidf_cos: float = 0.0,
) -> Dict[str, float]:
    """Compute a feature dict for one (S1, S2/S3) candidate pair."""
    n1 = s1_row["name_norm"]
    n2 = s23_row["name_norm"]
    a1 = s1_row["addr_norm"]
    a2 = s23_row["addr_norm"]

    # ── Name features ────────────────────────────────────────
    name_feats = {
        "name_tfidf_cos":     tfidf_cos,
        "name_lev":           Lev.normalized_similarity(n1, n2),
        "name_jaro":          JW.similarity(n1, n2),
        "name_token_sort":    fuzz.token_sort_ratio(n1, n2) / 100.0,
        "name_token_set":     fuzz.token_set_ratio(n1, n2) / 100.0,
        "name_partial":       fuzz.partial_ratio(n1, n2) / 100.0,
        "name_jaccard_3g":    _jaccard(_char_ngrams(n1, 3), _char_ngrams(n2, 3)),
        "name_jaccard_word":  _jaccard(_word_set(n1), _word_set(n2)),
        "name_common_words":  len(_word_set(n1) & _word_set(n2)),
        "name_len_ratio":     _len_ratio(n1, n2),
    }

    # ── Address features ─────────────────────────────────────
    if a1 == "" and a2 == "":
        addr_feats = {
            "addr_lev": 1.0, "addr_jaro": 1.0,
            "addr_token_sort": 1.0, "addr_jaccard_word": 1.0,
            "addr_common_words": 0, "addr_len_ratio": 1.0,
            "addr_either_empty": 1.0,
        }
    elif a1 == "" or a2 == "":
        addr_feats = {
            "addr_lev": 0.0, "addr_jaro": 0.0,
            "addr_token_sort": 0.0, "addr_jaccard_word": 0.0,
            "addr_common_words": 0, "addr_len_ratio": 0.0,
            "addr_either_empty": 1.0,
        }
    else:
        addr_feats = {
            "addr_lev":           Lev.normalized_similarity(a1, a2),
            "addr_jaro":          JW.similarity(a1, a2),
            "addr_token_sort":    fuzz.token_sort_ratio(a1, a2) / 100.0,
            "addr_jaccard_word":  _jaccard(_word_set(a1), _word_set(a2)),
            "addr_common_words":  len(_word_set(a1) & _word_set(a2)),
            "addr_len_ratio":     _len_ratio(a1, a2),
            "addr_either_empty":  0.0,
        }

    # ── Source indicator ─────────────────────────────────────
    source_feat = {"is_source3": float(s23_row["entity_id"].startswith("S3-"))}

    return {**name_feats, **addr_feats, **source_feat}


FEATURE_NAMES = list(compute_pair_features(
    pd.Series({"name_norm": "", "addr_norm": ""}),
    pd.Series({"name_norm": "", "addr_norm": "", "entity_id": "S2-0"}),
).keys())

print(f"✅ {len(FEATURE_NAMES)} features defined: {FEATURE_NAMES}")

# ================================================================
# TRAINING DATA BUILDER (for one country at a time)
# ================================================================

def build_training_pairs_for_country(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    gt: Dict[str, Set[str]],
    candidates: Dict[str, List[Tuple[str, float]]],
    neg_pos_ratio: int = CFG.NEG_POS_RATIO,
) -> Tuple[List[list], List[int], List[Tuple[str, str]]]:
    """
    Build (feature_rows, labels, pair_ids) for one country.
    Returns lists instead of numpy arrays so they can be aggregated.
    """
    s23_index = {row["entity_id"]: row for _, row in s23_df.iterrows()}
    s1_index  = {row["entity_id"]: row for _, row in s1_df.iterrows()}

    rows, labels, pair_ids = [], [], []

    for s1_id, cands in tqdm(candidates.items(), desc="    building pairs", leave=False):
        s1_row = s1_index.get(s1_id)
        if s1_row is None:
            continue
        true_matches = gt.get(s1_id, set())

        pos_pairs, neg_pairs = [], []
        for s23_id, cos_score in cands:
            s23_row = s23_index.get(s23_id)
            if s23_row is None:
                continue
            is_match = s23_id in true_matches
            feats = compute_pair_features(s1_row, s23_row, tfidf_cos=cos_score)
            feat_vec = [feats[f] for f in FEATURE_NAMES]
            if is_match:
                pos_pairs.append((feat_vec, s23_id))
            else:
                neg_pairs.append((feat_vec, s23_id))

        # Inject GT positives missed by blocking
        missed = true_matches - {c[0] for c in cands}
        for s23_id in missed:
            s23_row = s23_index.get(s23_id)
            if s23_row is None:
                continue
            feats = compute_pair_features(s1_row, s23_row, tfidf_cos=0.0)
            feat_vec = [feats[f] for f in FEATURE_NAMES]
            pos_pairs.append((feat_vec, s23_id))

        # Add all positives
        for fv, s23_id in pos_pairs:
            rows.append(fv)
            labels.append(1)
            pair_ids.append((s1_id, s23_id))

        # Sub-sample negatives
        n_neg = min(len(neg_pairs), max(neg_pos_ratio * len(pos_pairs), 1))
        if n_neg < len(neg_pairs):
            neg_idx = np.random.choice(len(neg_pairs), n_neg, replace=False)
            neg_pairs = [neg_pairs[i] for i in neg_idx]
        for fv, s23_id in neg_pairs:
            rows.append(fv)
            labels.append(0)
            pair_ids.append((s1_id, s23_id))

    return rows, labels, pair_ids


# ================================================================
# F_0.5 EVALUATION HELPERS
# ================================================================

def f_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> float:
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        if not true_set and not pred_set:
            scores.append(1.0)
        elif not pred_set or not true_set:
            scores.append(0.0)
        else:
            tp = len(pred_set & true_set)
            prec = tp / len(pred_set)
            rec  = tp / len(true_set)
            scores.append(f_beta_score(prec, rec, 0.5))
    return float(np.mean(scores))


def pairs_to_predictions(
    pair_ids: List[Tuple[str, str]],
    y_pred: np.ndarray,
    all_s1_ids: Set[str],
) -> Dict[str, Set[str]]:
    preds: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ids}
    for (s1_id, s23_id), label in zip(pair_ids, y_pred):
        if label == 1:
            preds[s1_id].add(s23_id)
    return preds


def find_best_threshold(
    pair_ids: List[Tuple[str, str]],
    y_proba: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: Set[str],
    thresholds: np.ndarray = CFG.THRESHOLD_GRID,
) -> Tuple[float, float]:
    best_t, best_f = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        preds  = pairs_to_predictions(pair_ids, y_pred, all_s1_ids)
        score  = macro_f05(preds, ground_truth)
        if score > best_f:
            best_t, best_f = float(t), score
    return best_t, best_f


print("✅ Evaluation helpers defined")

# ================================================================
#                       MAIN PIPELINE
# ================================================================

# ── Checkpoint directory ─────────────────────────────────────
CHECKPOINT_DIR = os.path.join(CFG.BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Checkpoint file paths
CKP_PHASE2 = os.path.join(CHECKPOINT_DIR, "phase2_features.pkl")
CKP_PHASE3 = os.path.join(CHECKPOINT_DIR, "phase3_model.pkl")
CKP_PHASE4_COUNTRY = lambda c: os.path.join(CHECKPOINT_DIR, f"phase4_{_safe_country(c)}.pkl")
CKP_PHASE4_DONE = os.path.join(CHECKPOINT_DIR, "phase4_done.pkl")


def checkpoint_exists(path):
    return os.path.isfile(path)


def save_checkpoint(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  💾 Checkpoint saved: {os.path.basename(path)}")


def load_checkpoint(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  ⏩ Checkpoint loaded: {os.path.basename(path)}")
    return data


# ── PHASE 1: Partition to disk ───────────────────────────────
train_part_dir = os.path.join(CFG.PARTITION_DIR, "train")
if os.path.isdir(train_part_dir) and len(glob.glob(os.path.join(train_part_dir, "s1_*.tsv"))) > 0:
    print("\n⏩ Partitions already exist — skipping Phase 1")
    print("   (Delete the 'partitions' folder to re-partition)")
    countries = get_countries("train")
    print(f"   🌐 Countries: {countries}")
else:
    countries = run_partitioning()

# ── Load ground truth (small, stays in memory) ───────────────
print("\n⏳ Loading ground truth …")
t0 = time.time()
gt_train = load_ground_truth()
print(f"  ✅ Ground truth loaded: {len(gt_train):,} entries ({time.time()-t0:.1f}s)")

# ================================================================
# PHASE 2: Country-by-country blocking + feature extraction
# ================================================================

if checkpoint_exists(CKP_PHASE2):
    print("\n" + "=" * 60)
    print("⏩ PHASE 2: Loading from checkpoint (skipping)")
    print("=" * 60)
    ckp2 = load_checkpoint(CKP_PHASE2)
    X_train     = ckp2["X_train"]
    y_train     = ckp2["y_train"]
    X_val       = ckp2["X_val"]
    y_val       = ckp2["y_val"]
    pairs_train = ckp2["pairs_train"]
    pairs_val   = ckp2["pairs_val"]
    s1_tr_set   = ckp2["s1_tr_set"]
    s1_va_set   = ckp2["s1_va_set"]
    all_pair_ids = ckp2["all_pair_ids"]
    overall_recall = ckp2["overall_recall"]
    del ckp2; gc.collect()
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Blocking recall: {overall_recall:.4f}")
else:
    print("\n" + "=" * 60)
    print("PHASE 2: Country-by-country blocking & feature extraction")
    print("=" * 60)

    all_rows, all_labels, all_pair_ids = [], [], []
    total_block_captured, total_block_total = 0, 0

    for country in countries:
        print(f"\n  ╔══ 🌐 Country: {country} ══════════════════════════════")

        s1_c = load_partition("train", "s1", country)
        s23_c = load_partition("train", "s23", country)

        if s1_c is None or s23_c is None:
            print(f"  ║  ⚠️  Missing partition files — skipping")
            continue

        print(f"  ║  S1={len(s1_c):,}   S23={len(s23_c):,}")

        # Sample a fraction of S1 entities for training
        n_sample = max(1, int(len(s1_c) * CFG.TRAIN_SAMPLE_FRAC))
        sample_ids = np.random.choice(
            s1_c["entity_id"].values, size=n_sample, replace=False)
        s1_sample = s1_c[s1_c["entity_id"].isin(set(sample_ids))].reset_index(drop=True)
        print(f"  ║  Sampled {len(s1_sample):,} S1 entities")

        # Run blocking
        t0 = time.time()
        cands = run_blocking_country(s1_sample, s23_c)
        print(f"  ║  Blocking: {(time.time()-t0)/60:.1f} min")

        # Measure blocking recall
        n_cap, n_tot = 0, 0
        for s1_id in sample_ids:
            true = gt_train.get(s1_id, set())
            cand_ids = {c[0] for c in cands.get(s1_id, [])}
            n_cap += len(true & cand_ids)
            n_tot += len(true)
        total_block_captured += n_cap
        total_block_total += n_tot
        recall = n_cap / max(n_tot, 1)
        print(f"  ║  Blocking recall: {recall:.4f} ({n_cap}/{n_tot})")

        # Build features
        t0 = time.time()
        rows, labels, pids = build_training_pairs_for_country(
            s1_sample, s23_c, gt_train, cands)
        print(f"  ║  Features: {len(rows):,} pairs in {(time.time()-t0)/60:.1f} min")
        print(f"  ╚══════════════════════════════════════════════════════")

        all_rows.extend(rows)
        all_labels.extend(labels)
        all_pair_ids.extend(pids)

        # Free this country's data
        del s1_c, s23_c, s1_sample, cands, rows, labels, pids
        gc.collect()

    # Convert to numpy
    X_all = np.array(all_rows, dtype=np.float32)
    y_all = np.array(all_labels, dtype=np.int32)
    del all_rows, all_labels; gc.collect()

    overall_recall = total_block_captured / max(total_block_total, 1)
    print(f"\n  📊 Overall blocking recall: {overall_recall:.4f}")
    print(f"  📊 Total training pairs: {len(X_all):,}  "
          f"(pos={y_all.sum():,}  neg={(y_all==0).sum():,})")

    # ── Train / validation split (by S1 entity) ──────────────
    unique_s1 = list({p[0] for p in all_pair_ids})
    s1_tr, s1_va = train_test_split(unique_s1, test_size=0.2, random_state=CFG.RANDOM_SEED)
    s1_tr_set, s1_va_set = set(s1_tr), set(s1_va)

    tr_mask = np.array([p[0] in s1_tr_set for p in all_pair_ids])
    va_mask = ~tr_mask

    X_train = X_all[tr_mask]
    y_train = y_all[tr_mask]
    pairs_train = [p for p, m in zip(all_pair_ids, tr_mask) if m]

    X_val = X_all[va_mask]
    y_val = y_all[va_mask]
    pairs_val = [p for p, m in zip(all_pair_ids, va_mask) if m]

    print(f"  Train split: {len(X_train):,} | Val split: {len(X_val):,}")

    del X_all, y_all; gc.collect()

    # ── SAVE CHECKPOINT ──────────────────────────────────────
    save_checkpoint(CKP_PHASE2, {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "pairs_train": pairs_train, "pairs_val": pairs_val,
        "s1_tr_set": s1_tr_set, "s1_va_set": s1_va_set,
        "all_pair_ids": all_pair_ids,
        "overall_recall": overall_recall,
    })

# ================================================================
# PHASE 3: HYPERPARAMETER TUNING + FINAL MODEL
# ================================================================

if checkpoint_exists(CKP_PHASE3):
    print("\n" + "=" * 60)
    print("⏩ PHASE 3: Loading model from checkpoint (skipping)")
    print("=" * 60)
    ckp3 = load_checkpoint(CKP_PHASE3)
    final_model    = ckp3["final_model"]
    best_threshold = ckp3["best_threshold"]
    best_params    = ckp3["best_params"]
    val_f05        = ckp3["val_f05"]
    del ckp3; gc.collect()
    print(f"  🎯 Threshold: {best_threshold:.2f} | Val F_0.5: {val_f05:.4f}")

    # Free training data — no longer needed
    del X_train, y_train, X_val, y_val; gc.collect()
else:
    print("\n" + "=" * 60)
    print("PHASE 3: Hyperparameter Tuning with Optuna")
    print("=" * 60)

    gt_val = {s1: gt_train.get(s1, set()) for s1 in s1_va_set}

    def optuna_objective(trial: optuna.Trial) -> float:
        params = {
            "objective":        "binary",
            "metric":           "binary_logloss",
            "verbosity":        -1,
            "n_jobs":           CFG.N_JOBS,
            "random_state":     CFG.RANDOM_SEED,
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "num_leaves":       trial.suggest_int("num_leaves", 15, 256),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        y_proba = model.predict_proba(X_val)[:, 1]
        _, best_f05 = find_best_threshold(pairs_val, y_proba, gt_val, s1_va_set)
        return best_f05

    print(f"  Running {CFG.OPTUNA_TRIALS} trials (timeout={CFG.OPTUNA_TIMEOUT}s) …")
    t0 = time.time()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=CFG.RANDOM_SEED),
    )
    study.optimize(
        optuna_objective,
        n_trials=CFG.OPTUNA_TRIALS,
        timeout=CFG.OPTUNA_TIMEOUT,
        show_progress_bar=True,
    )

    elapsed = time.time() - t0
    print(f"\n  ⏱ Tuning completed in {elapsed/60:.1f} min")
    print(f"  🏆 Best F_0.5 = {study.best_value:.4f}")
    print(f"  📋 Best params:")
    for k, v in study.best_params.items():
        print(f"     {k}: {v}")

    best_params = study.best_params

    # ── Train final model ────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3b: Training final model with best hyperparameters")
    print("=" * 60)

    final_params = {
        "objective":        "binary",
        "metric":           "binary_logloss",
        "verbosity":        -1,
        "n_jobs":           CFG.N_JOBS,
        "random_state":     CFG.RANDOM_SEED,
        "n_estimators":     CFG.FINAL_N_ESTIMATORS,
        **best_params,
    }
    final_params["n_estimators"] = CFG.FINAL_N_ESTIMATORS

    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(CFG.EARLY_STOP_ROUNDS, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    # Find optimal threshold
    y_val_proba = final_model.predict_proba(X_val)[:, 1]
    best_threshold, val_f05 = find_best_threshold(
        pairs_val, y_val_proba, gt_val, s1_va_set,
    )

    print(f"\n  ✅ Final model trained (best_iteration={final_model.best_iteration_})")
    print(f"  🎯 Optimal threshold = {best_threshold:.2f}")
    print(f"  📊 Validation F_0.5  = {val_f05:.4f}")

    # Feature importance
    imp = sorted(zip(FEATURE_NAMES, final_model.feature_importances_),
                 key=lambda x: -x[1])
    print("\n  📊 Feature Importance (top 10):")
    for name, score in imp[:10]:
        print(f"     {name:25s} {score:6d}")

    # Free training arrays
    del X_train, y_train, X_val, y_val
    gc.collect()

    # ── SAVE CHECKPOINT ──────────────────────────────────────
    save_checkpoint(CKP_PHASE3, {
        "final_model":    final_model,
        "best_threshold": best_threshold,
        "best_params":    best_params,
        "val_f05":        val_f05,
    })

# ================================================================
# PHASE 4: COUNTRY-BY-COUNTRY TEST INFERENCE
# ================================================================
print("\n" + "=" * 60)
print("PHASE 4: Country-by-country test inference")
print("=" * 60)

test_countries = get_countries("test")
print(f"  🌐 Test countries: {test_countries}")

matching_results:  Dict[str, Set[str]] = {}
candidate_results: Dict[str, Set[str]] = {}

# Check if Phase 4 is already fully done
if checkpoint_exists(CKP_PHASE4_DONE):
    print("\n⏩ Phase 4 already complete — loading results")
    ckp4 = load_checkpoint(CKP_PHASE4_DONE)
    matching_results  = ckp4["matching_results"]
    candidate_results = ckp4["candidate_results"]
    del ckp4; gc.collect()
else:
    for country in test_countries:
        ckp_path = CKP_PHASE4_COUNTRY(country)

        # Check if this country was already processed
        if checkpoint_exists(ckp_path):
            print(f"\n  ⏩ Country '{country}' already processed — loading checkpoint")
            ckp_c = load_checkpoint(ckp_path)
            matching_results.update(ckp_c["matching"])
            candidate_results.update(ckp_c["candidates"])
            del ckp_c; gc.collect()
            continue

        print(f"\n  ╔══ 🌐 Test Country: {country} ═══════════════════════")

        s1_c = load_partition("test", "s1", country)
        s23_c = load_partition("test", "s23", country)

        if s1_c is None:
            print(f"  ║  ⚠️  No S1 test data — skipping")
            continue
        if s23_c is None:
            print(f"  ║  ⚠️  No S23 test data — all {len(s1_c):,} are singletons")
            country_matching = {}
            country_candidates = {}
            for s1_id in s1_c["entity_id"].values:
                country_matching[s1_id] = set()
                country_candidates[s1_id] = set()
            matching_results.update(country_matching)
            candidate_results.update(country_candidates)
            save_checkpoint(ckp_path, {
                "matching": country_matching, "candidates": country_candidates})
            del s1_c; gc.collect()
            continue

        print(f"  ║  S1={len(s1_c):,}   S23={len(s23_c):,}")

        # Run blocking
        t0 = time.time()
        test_cands = run_blocking_country(s1_c, s23_c)
        print(f"  ║  Blocking: {(time.time()-t0)/60:.1f} min")

        # Build index for fast lookup
        s23_idx = {row["entity_id"]: row for _, row in s23_c.iterrows()}
        s1_idx  = {row["entity_id"]: row for _, row in s1_c.iterrows()}

        # Predict
        country_matching = {}
        country_candidates = {}

        for s1_id in tqdm(s1_c["entity_id"].values,
                          desc="    predicting", leave=False):
            cands = test_cands.get(s1_id, [])
            cand_set = {c[0] for c in cands}
            country_candidates[s1_id] = cand_set

            if not cands:
                country_matching[s1_id] = set()
                continue

            s1_row = s1_idx[s1_id]
            feat_rows, cand_ids = [], []
            for s23_id, cos_score in cands:
                s23_row = s23_idx.get(s23_id)
                if s23_row is None:
                    continue
                feats = compute_pair_features(s1_row, s23_row, tfidf_cos=cos_score)
                feat_rows.append([feats[f] for f in FEATURE_NAMES])
                cand_ids.append(s23_id)

            if not feat_rows:
                country_matching[s1_id] = set()
                continue

            X_pred = np.array(feat_rows, dtype=np.float32)
            proba  = final_model.predict_proba(X_pred)[:, 1]

            matches = set()
            for idx, p in enumerate(proba):
                if p >= best_threshold:
                    matches.add(cand_ids[idx])
            country_matching[s1_id] = matches

        n_matched = sum(1 for v in country_matching.values() if v)
        print(f"  ║  Matched: {n_matched:,} / {len(s1_c):,}")
        print(f"  ╚══════════════════════════════════════════════════════")

        # Save per-country checkpoint
        matching_results.update(country_matching)
        candidate_results.update(country_candidates)
        save_checkpoint(ckp_path, {
            "matching": country_matching, "candidates": country_candidates})

        del s1_c, s23_c, test_cands, s23_idx, s1_idx
        del country_matching, country_candidates; gc.collect()

    # Save Phase 4 completion marker
    save_checkpoint(CKP_PHASE4_DONE, {
        "matching_results": matching_results,
        "candidate_results": candidate_results,
    })

print(f"\n  ✅ Total predictions: {len(matching_results):,}")
n_matched = sum(1 for v in matching_results.values() if v)
n_singleton = sum(1 for v in matching_results.values() if not v)
print(f"  📊 Matched: {n_matched:,} | Singletons: {n_singleton:,}")

# ================================================================
# PHASE 5: SAVE OUTPUT FILES
# ================================================================
print("\n" + "=" * 60)
print("PHASE 5: Saving output files")
print("=" * 60)

# Collect all test S1 IDs in order
all_test_s1_ids = []
for country in test_countries:
    s1_c = load_partition("test", "s1", country)
    if s1_c is not None:
        all_test_s1_ids.extend(s1_c["entity_id"].values.tolist())
        del s1_c; gc.collect()

# ── matching_results.tsv ─────────────────────────────────────
matching_path = os.path.join(CFG.OUTPUT_DIR, "matching_results.tsv")
with open(matching_path, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s1_id in all_test_s1_ids:
        matches = matching_results.get(s1_id, set())
        ids_str = ",".join(sorted(matches)) if matches else ""
        f.write(f"{s1_id}\t{ids_str}\n")

print(f"  ✅ Saved: {matching_path}")

# ── candidate_pairs.tsv ──────────────────────────────────────
candidate_path = os.path.join(CFG.OUTPUT_DIR, "candidate_pairs.tsv")
with open(candidate_path, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1_id in all_test_s1_ids:
        cands = candidate_results.get(s1_id, set())
        ids_str = ",".join(sorted(cands)) if cands else ""
        f.write(f"{s1_id}\t{ids_str}\n")

print(f"  ✅ Saved: {candidate_path}")

# ── Validate submission ──────────────────────────────────────
print("\n" + "=" * 60)
print("Validating submission files")
print("=" * 60)

validate_script = os.path.join(CFG.BASE_DIR, "utils", "validate_submission.py")
if os.path.isfile(validate_script):
    import subprocess
    result = subprocess.run(
        [
            sys.executable, validate_script,
            "--matching", matching_path,
            "--candidate", candidate_path,
            "--test-dir", CFG.TEST_DIR,
        ],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
else:
    print("  ⚠️ validate_submission.py not found — skipping validation")

# ── Save model & config ─────────────────────────────────────
model_path = os.path.join(CFG.OUTPUT_DIR, "final_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(final_model, f)
print(f"  ✅ Model saved: {model_path}")

config_out = {
    "best_threshold":     best_threshold,
    "val_f05":            val_f05,
    "best_params":        best_params,
    "feature_names":      FEATURE_NAMES,
    "blocking_recall":    overall_recall,
    "n_train_pairs":      int(len(pairs_train)),
    "n_val_pairs":        int(len(pairs_val)),
}
config_path = os.path.join(CFG.OUTPUT_DIR, "run_config.json")
with open(config_path, "w") as f:
    json.dump(config_out, f, indent=2, default=str)
print(f"  ✅ Config saved: {config_path}")

# ── DONE ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🎉  PIPELINE COMPLETE!")
print("=" * 60)
print(f"  📁 Output directory: {CFG.OUTPUT_DIR}")
print(f"  📄 matching_results.tsv  — upload to leaderboard")
print(f"  📄 candidate_pairs.tsv   — include in submission zip")
print(f"  🎯 Threshold: {best_threshold:.2f}  |  Val F_0.5: {val_f05:.4f}")
print(f"\n  💡 To start fresh, delete the 'checkpoints' folder:")
print(f"     rm -rf {CHECKPOINT_DIR}")

