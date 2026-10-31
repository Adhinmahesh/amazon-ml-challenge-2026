#!/usr/bin/env python3
# %% [markdown]
# # 🏢 Amazon ML Challenge 2026 — Business Entity Resolution
# ## Complete Solution for Google Colab
#
# **Pipeline Overview:**
# 1. Load & preprocess data (normalize names/addresses)
# 2. Blocking / Candidate Generation (Country + TF-IDF char n-grams)
# 3. Feature Engineering (string similarity features)
# 4. Hyperparameter Tuning with Optuna
# 5. Train LightGBM classifier
# 6. Threshold optimisation for F₀.₅
# 7. Full test-set inference & output generation

# %% ============================================================
# CELL 1 — INSTALL DEPENDENCIES
# ============================================================
# COLAB:  !pip install -q sparse-dot-topn rapidfuzz lightgbm optuna tqdm
# LOCAL:  python3 -m venv venv && venv/bin/pip install numpy pandas scikit-learn \
#         sparse-dot-topn rapidfuzz lightgbm optuna tqdm
# Then run with:  venv/bin/python solution_colab.py

# %% ============================================================
# CELL 2 — IMPORTS
# ============================================================
import os, gc, re, sys, time, warnings, math
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

# %% ============================================================
# CELL 3 — CONFIGURATION
# ============================================================
class Config:
    """Central configuration — edit paths & hyper-params here."""

    # ── Paths ────────────────────────────────────────────────
    # LOCAL: points to the student_resource directory
    # COLAB: change to "/content" or wherever you uploaded data
    BASE_DIR   = "/home/user/Desktop/haws/6ab10eb3b23ba_student_resource/student_resource"
    TRAIN_DIR  = os.path.join(BASE_DIR, "dataset", "train")
    TEST_DIR   = os.path.join(BASE_DIR, "dataset", "test")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    # ── Blocking ─────────────────────────────────────────────
    # Reduced TOP_K and chunk size for 15 GB RAM machines
    TOP_K              = 20      # candidates per S1 entity
    TFIDF_MIN_SIM      = 0.05   # minimum cosine sim to keep
    TFIDF_NGRAM_RANGE  = (3, 4) # character n-gram range
    BLOCK_CHUNK_SIZE   = 1_000  # S1 entities per blocking batch

    # ── Sampling (training / tuning) ─────────────────────────
    # Reduced for 15 GB RAM; increase if you have more memory
    TRAIN_SAMPLE_S1    = 20_000   # S1 entities for classifier training
    TUNE_SAMPLE_S1     = 5_000   # S1 entities for Optuna tuning
    NEG_POS_RATIO      = 3        # negative : positive ratio cap

    # ── Optuna ───────────────────────────────────────────────
    OPTUNA_TRIALS      = 25
    OPTUNA_TIMEOUT     = 1800     # seconds (30 min)

    # ── Final model ──────────────────────────────────────────
    FINAL_N_ESTIMATORS = 2000
    EARLY_STOP_ROUNDS  = 100
    THRESHOLD_GRID     = np.arange(0.10, 0.96, 0.02)

    # ── Misc ─────────────────────────────────────────────────
    RANDOM_SEED        = 42
    N_JOBS             = -1       # LightGBM parallelism

CFG = Config()
os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
np.random.seed(CFG.RANDOM_SEED)

print("✅ Configuration loaded")

# %% ============================================================
# CELL 4 — TEXT NORMALISATION HELPERS
# ============================================================

# ── Business-name abbreviation expansions ────────────────────
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

# ── Address abbreviation expansions ──────────────────────────
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

# Precompiled patterns
_RE_AMP      = re.compile(r"&")
_RE_NON_ALNUM = re.compile(r"[^\w\s]", re.UNICODE)
_RE_MULTI_WS = re.compile(r"\s+")
_RE_LEADING  = re.compile(r"^[^a-zA-Z0-9\u0900-\u097F\u00C0-\u024F]+")


def _base_normalise(text: str) -> str:
    """Shared low-level normalisation."""
    if pd.isna(text) or not str(text).strip():
        return ""
    text = str(text).lower().strip()
    text = _RE_LEADING.sub("", text)
    text = _RE_AMP.sub(" and ", text)
    text = _RE_NON_ALNUM.sub(" ", text)
    text = _RE_MULTI_WS.sub(" ", text).strip()
    return text


def normalise_name(text: str) -> str:
    """Normalise a business name."""
    text = _base_normalise(text)
    if not text:
        return ""
    for pat, repl in _NAME_SUBS:
        text = pat.sub(repl, text)
    text = _RE_MULTI_WS.sub(" ", text).strip()
    return text


def normalise_address(text: str) -> str:
    """Normalise a business address."""
    text = _base_normalise(text)
    if not text:
        return ""
    for pat, repl in _ADDR_SUBS:
        text = pat.sub(repl, text)
    text = _RE_MULTI_WS.sub(" ", text).strip()
    return text


print("✅ Normalisation functions defined")

# %% ============================================================
# CELL 5 — DATA LOADING & PREPROCESSING
# ============================================================

def load_source(path: str) -> pd.DataFrame:
    """Load a source TSV and add normalised columns."""
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", dtype=str, na_filter=False)
    df["name_norm"]  = df["business_name"].apply(normalise_name)
    df["addr_norm"]  = df["business_address"].apply(normalise_address)
    # Combined text for TF-IDF blocking
    df["combined"]   = df["name_norm"] + " " + df["addr_norm"]
    elapsed = time.time() - t0
    print(f"  Loaded {os.path.basename(path)}: {len(df):,} rows  ({elapsed:.1f}s)")
    return df


def load_ground_truth(path: str) -> Dict[str, Set[str]]:
    """Return {s1_id: set(matched_ids)} from ground truth TSV."""
    gt = pd.read_csv(path, sep="\t", dtype=str, na_filter=False)
    mapping: Dict[str, Set[str]] = {}
    for _, row in gt.iterrows():
        s1 = row["source1_entity_id"]
        ids_str = row["matched_entity_ids"]
        if ids_str.strip():
            mapping[s1] = set(ids_str.split(","))
        else:
            mapping[s1] = set()
    return mapping


print("⏳ Loading training data (this may take a few minutes) …")

s1_train = load_source(os.path.join(CFG.TRAIN_DIR, "train_source1.tsv"))
s2_train = load_source(os.path.join(CFG.TRAIN_DIR, "train_source2.tsv"))
s3_train = load_source(os.path.join(CFG.TRAIN_DIR, "train_source3.tsv"))

print("⏳ Loading ground truth …")
gt_train = load_ground_truth(os.path.join(CFG.TRAIN_DIR, "train_ground_truth.tsv"))

# Merge S2 + S3 for candidate search
s23_train = pd.concat([s2_train, s3_train], ignore_index=True)
del s2_train, s3_train; gc.collect()

print(f"\n✅ Training data ready: "
      f"S1={len(s1_train):,}  S2+S3={len(s23_train):,}  "
      f"GT entries={len(gt_train):,}")

# %% ============================================================
# CELL 6 — BLOCKING  (Country + TF-IDF char n-gram)
# ============================================================

def build_blocker_for_country(
    s23_country: pd.DataFrame,
    ngram_range: Tuple[int, int] = CFG.TFIDF_NGRAM_RANGE,
):
    """Fit a character n-gram TF-IDF on S2+S3 entities for one country."""
    gc.collect() # Free up memory before creating sparse matrix
    vectoriser = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 3),   # Restrict to trigrams only for memory
        max_features=30_000,  # Drastically reduced features
        min_df=5,             # Ignore very rare character strings
        sublinear_tf=True,
        dtype=np.float32,
    )
    tfidf_s23 = vectoriser.fit_transform(s23_country["combined"])
    gc.collect()
    return vectoriser, tfidf_s23


def run_blocking_country(
    s1_country: pd.DataFrame,
    s23_country: pd.DataFrame,
    top_k: int = CFG.TOP_K,
    min_sim: float = CFG.TFIDF_MIN_SIM,
    chunk_size: int = CFG.BLOCK_CHUNK_SIZE,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    For every S1 entity in one country, find top-K candidate S2/S3
    entities by TF-IDF cosine similarity.

    Returns {s1_id: [(s23_id, cosine_score), ...]}
    """
    if len(s1_country) == 0 or len(s23_country) == 0:
        return {}

    vectoriser, tfidf_s23 = build_blocker_for_country(s23_country)
    s23_ids = s23_country["entity_id"].values

    candidates: Dict[str, List[Tuple[str, float]]] = {}
    n_chunks = math.ceil(len(s1_country) / chunk_size)

    for i in tqdm(range(n_chunks), desc="  blocking chunks", leave=False):
        start = i * chunk_size
        end   = min(start + chunk_size, len(s1_country))
        chunk = s1_country.iloc[start:end]

        tfidf_s1 = vectoriser.transform(chunk["combined"])
        sim_matrix = awesome_cossim_topn(
            tfidf_s1, tfidf_s23.T,
            ntop=top_k,
            lower_bound=min_sim,
            use_threads=True,
            n_jobs=4,
        )

        # Extract per-row results from sparse matrix
        coo = sim_matrix.tocoo()
        s1_chunk_ids = chunk["entity_id"].values
        for row, col, score in zip(coo.row, coo.col, coo.data):
            s1_id  = s1_chunk_ids[row]
            s23_id = s23_ids[col]
            candidates.setdefault(s1_id, []).append((s23_id, float(score)))

    # Also ensure every S1 entity gets an entry (even if empty)
    for s1_id in s1_country["entity_id"].values:
        candidates.setdefault(s1_id, [])

    return candidates


def run_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    **kwargs,
) -> Dict[str, List[Tuple[str, float]]]:
    """Run blocking across all countries (partitioned by country)."""
    all_candidates: Dict[str, List[Tuple[str, float]]] = {}
    countries = sorted(s1_df["country"].unique())

    for country in countries:
        s1_c  = s1_df[s1_df["country"] == country].reset_index(drop=True)
        s23_c = s23_df[s23_df["country"] == country].reset_index(drop=True)
        print(f"  🌐 Country={country}: S1={len(s1_c):,}  S23={len(s23_c):,}")

        cands = run_blocking_country(s1_c, s23_c, **kwargs)
        all_candidates.update(cands)

        del s1_c, s23_c, cands; gc.collect()

    # Ensure all S1 entities have an entry
    for s1_id in s1_df["entity_id"].values:
        all_candidates.setdefault(s1_id, [])

    return all_candidates


print("✅ Blocking functions defined")

# %% ============================================================
# CELL 7 — FEATURE ENGINEERING
# ============================================================

def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Return the set of character n-grams."""
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
    addr_either_empty = float(a1 == "" or a2 == "")
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

# %% ============================================================
# CELL 8 — BUILD TRAINING DATA
# ============================================================

def build_training_pairs(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    gt: Dict[str, Set[str]],
    candidates: Dict[str, List[Tuple[str, float]]],
    neg_pos_ratio: int = CFG.NEG_POS_RATIO,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Build feature matrix X, label vector y, and pair-id list from
    blocking candidates + ground truth.

    Positives: pairs that appear in both candidates and ground truth.
    Hard negatives: candidate pairs NOT in ground truth (capped by ratio).
    Also injects GT positives missed by blocking for better training.
    """
    # Index S23 for fast lookup
    s23_index = {row["entity_id"]: row for _, row in s23_df.iterrows()}
    s1_index  = {row["entity_id"]: row for _, row in s1_df.iterrows()}

    rows, labels, pair_ids = [], [], []

    for s1_id, cands in tqdm(candidates.items(), desc="  building pairs", leave=False):
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

        # Inject ground-truth positives missed by blocking
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
            neg_indices = np.random.choice(len(neg_pairs), n_neg, replace=False)
            neg_pairs = [neg_pairs[i] for i in neg_indices]
        for fv, s23_id in neg_pairs:
            rows.append(fv)
            labels.append(0)
            pair_ids.append((s1_id, s23_id))

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"  Built training data: {len(X):,} pairs  "
          f"(pos={y.sum():,}  neg={(y==0).sum():,}  ratio=1:{(y==0).sum()/max(y.sum(),1):.1f})")
    return X, y, pair_ids


print("✅ Training data builder defined")

# %% ============================================================
# CELL 9 — F_0.5 EVALUATION HELPERS
# ============================================================

def f_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F_beta from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> float:
    """Compute macro-averaged F_0.5 over all S1 entities."""
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
    """Convert pair-level predictions to entity-level predictions."""
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
    """Grid-search for the threshold that maximises F_0.5."""
    best_t, best_f = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        preds  = pairs_to_predictions(pair_ids, y_pred, all_s1_ids)
        score  = macro_f05(preds, ground_truth)
        if score > best_f:
            best_t, best_f = float(t), score
    return best_t, best_f


print("✅ Evaluation helpers defined")

# %% ============================================================
# CELL 10 — SAMPLE, BLOCK, AND BUILD TRAINING SET
# ============================================================
print("=" * 60)
print("STEP 1: Sampling S1 entities for training")
print("=" * 60)

# Sample S1 entities for training
all_s1_ids_train = s1_train["entity_id"].values
sample_ids = np.random.choice(
    all_s1_ids_train,
    size=min(CFG.TRAIN_SAMPLE_S1, len(all_s1_ids_train)),
    replace=False,
)
s1_sample = s1_train[s1_train["entity_id"].isin(set(sample_ids))].reset_index(drop=True)
print(f"  Sampled {len(s1_sample):,} S1 entities for training")

# ── Run blocking on the sample ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Running blocking on training sample")
print("=" * 60)

t0 = time.time()
train_candidates = run_blocking(s1_sample, s23_train)
print(f"  ⏱ Blocking completed in {(time.time()-t0)/60:.1f} min")

# Measure blocking recall
n_captured, n_total = 0, 0
for s1_id in sample_ids:
    true = gt_train.get(s1_id, set())
    cand_ids = {c[0] for c in train_candidates.get(s1_id, [])}
    n_captured += len(true & cand_ids)
    n_total    += len(true)
block_recall = n_captured / max(n_total, 1)
print(f"  📊 Blocking recall on sample: {block_recall:.4f}  "
      f"({n_captured:,}/{n_total:,} true matches found)")

# ── Build feature matrix ─────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Computing features for training pairs")
print("=" * 60)

t0 = time.time()
X_all, y_all, pairs_all = build_training_pairs(
    s1_sample, s23_train, gt_train, train_candidates,
)
print(f"  ⏱ Feature extraction completed in {(time.time()-t0)/60:.1f} min")

# Split into train / validation (by S1 entity to avoid leakage)
unique_s1 = list({p[0] for p in pairs_all})
s1_tr, s1_va = train_test_split(unique_s1, test_size=0.2, random_state=CFG.RANDOM_SEED)
s1_tr_set, s1_va_set = set(s1_tr), set(s1_va)

tr_mask = np.array([p[0] in s1_tr_set for p in pairs_all])
va_mask = ~tr_mask

X_train, y_train, pairs_train = X_all[tr_mask], y_all[tr_mask], [p for p, m in zip(pairs_all, tr_mask) if m]
X_val,   y_val,   pairs_val   = X_all[va_mask], y_all[va_mask], [p for p, m in zip(pairs_all, va_mask) if m]

print(f"  Train split: {len(X_train):,} pairs | Val split: {len(X_val):,} pairs")

del X_all, y_all, pairs_all; gc.collect()

# %% ============================================================
# CELL 11 — HYPERPARAMETER TUNING WITH OPTUNA
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Hyperparameter Tuning with Optuna")
print("=" * 60)

# For speed, tune on a smaller sub-sample of the training data
tune_size = min(CFG.TUNE_SAMPLE_S1 * 10, len(X_train))
tune_idx  = np.random.choice(len(X_train), tune_size, replace=False)
X_tune, y_tune = X_train[tune_idx], y_train[tune_idx]

# Build a sub-GT for validation entities
gt_val = {s1: gt_train.get(s1, set()) for s1 in s1_va_set}


def optuna_objective(trial: optuna.Trial) -> float:
    """Optuna objective: train LightGBM, evaluate F_0.5."""
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
        X_tune, y_tune,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    y_proba = model.predict_proba(X_val)[:, 1]
    _, best_f05 = find_best_threshold(
        pairs_val, y_proba, gt_val, s1_va_set,
    )
    return best_f05


print(f"  Running {CFG.OPTUNA_TRIALS} Optuna trials "
      f"(timeout={CFG.OPTUNA_TIMEOUT}s) …")
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
del X_tune, y_tune; gc.collect()

# %% ============================================================
# CELL 12 — TRAIN FINAL MODEL WITH BEST PARAMS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Training final model with best hyperparameters")
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
# Override n_estimators to use more with early stopping
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

# ── Find optimal threshold ───────────────────────────────────
y_val_proba = final_model.predict_proba(X_val)[:, 1]
best_threshold, val_f05 = find_best_threshold(
    pairs_val, y_val_proba, gt_val, s1_va_set,
)

print(f"\n  ✅ Final model trained (best_iteration={final_model.best_iteration_})")
print(f"  🎯 Optimal threshold = {best_threshold:.2f}")
print(f"  📊 Validation F_0.5  = {val_f05:.4f}")

# ── Feature importance ───────────────────────────────────────
imp = sorted(zip(FEATURE_NAMES, final_model.feature_importances_),
             key=lambda x: -x[1])
print("\n  📊 Feature Importance (top 10):")
for name, score in imp[:10]:
    print(f"     {name:25s} {score:6d}")

# Clean up training data
del X_train, y_train, X_val, y_val
del train_candidates, s1_sample
gc.collect()

# %% ============================================================
# CELL 13 — LOAD TEST DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Loading test data")
print("=" * 60)

s1_test  = load_source(os.path.join(CFG.TEST_DIR, "test_source1.tsv"))
s2_test  = load_source(os.path.join(CFG.TEST_DIR, "test_source2.tsv"))
s3_test  = load_source(os.path.join(CFG.TEST_DIR, "test_source3.tsv"))

s23_test = pd.concat([s2_test, s3_test], ignore_index=True)
del s2_test, s3_test; gc.collect()

print(f"\n  ✅ Test data: S1={len(s1_test):,}  S2+S3={len(s23_test):,}")

# Free training S23 data (no longer needed)
del s23_train; gc.collect()

# %% ============================================================
# CELL 14 — RUN BLOCKING ON TEST DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Running blocking on full test data")
print("=" * 60)

t0 = time.time()
test_candidates = run_blocking(s1_test, s23_test)
print(f"  ⏱ Test blocking completed in {(time.time()-t0)/60:.1f} min")

total_cands = sum(len(v) for v in test_candidates.values())
print(f"  📊 Total candidate pairs: {total_cands:,}")
print(f"  📊 Average candidates/S1: {total_cands/max(len(test_candidates),1):.1f}")

# %% ============================================================
# CELL 15 — GENERATE TEST PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Computing features & predicting on test data")
print("=" * 60)

# Build index for fast lookup
s23_test_index = {}
for _, row in tqdm(s23_test.iterrows(), total=len(s23_test),
                   desc="  indexing S23", leave=False):
    s23_test_index[row["entity_id"]] = row

s1_test_index = {}
for _, row in tqdm(s1_test.iterrows(), total=len(s1_test),
                   desc="  indexing S1", leave=False):
    s1_test_index[row["entity_id"]] = row


# Process predictions in chunks to manage memory
PRED_CHUNK = 10_000
s1_test_ids = s1_test["entity_id"].values

matching_results:   Dict[str, Set[str]]  = {}   # final matches
candidate_results:  Dict[str, Set[str]]  = {}   # all blocking candidates

n_chunks = math.ceil(len(s1_test_ids) / PRED_CHUNK)

for chunk_i in tqdm(range(n_chunks), desc="  predicting"):
    start = chunk_i * PRED_CHUNK
    end   = min(start + PRED_CHUNK, len(s1_test_ids))
    chunk_s1_ids = s1_test_ids[start:end]

    for s1_id in chunk_s1_ids:
        cands = test_candidates.get(s1_id, [])

        # Record all candidates for candidate_pairs.tsv
        cand_set = {c[0] for c in cands}
        candidate_results[s1_id] = cand_set

        if not cands:
            matching_results[s1_id] = set()
            continue

        # Compute features for all candidates of this S1
        s1_row = s1_test_index[s1_id]
        feat_rows = []
        cand_ids   = []
        for s23_id, cos_score in cands:
            s23_row = s23_test_index.get(s23_id)
            if s23_row is None:
                continue
            feats = compute_pair_features(s1_row, s23_row, tfidf_cos=cos_score)
            feat_rows.append([feats[f] for f in FEATURE_NAMES])
            cand_ids.append(s23_id)

        if not feat_rows:
            matching_results[s1_id] = set()
            continue

        X_pred = np.array(feat_rows, dtype=np.float32)
        proba  = final_model.predict_proba(X_pred)[:, 1]

        matches = set()
        for idx, p in enumerate(proba):
            if p >= best_threshold:
                matches.add(cand_ids[idx])
        matching_results[s1_id] = matches

print(f"\n  ✅ Predictions generated for {len(matching_results):,} S1 entities")
n_matched = sum(1 for v in matching_results.values() if v)
n_singleton = sum(1 for v in matching_results.values() if not v)
print(f"  📊 Matched: {n_matched:,} | Singletons: {n_singleton:,}")

# %% ============================================================
# CELL 16 — SAVE OUTPUT FILES
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Saving output files")
print("=" * 60)

# ── matching_results.tsv ─────────────────────────────────────
matching_path = os.path.join(CFG.OUTPUT_DIR, "matching_results.tsv")
with open(matching_path, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s1_id in s1_test["entity_id"].values:
        matches = matching_results.get(s1_id, set())
        ids_str = ",".join(sorted(matches)) if matches else ""
        f.write(f"{s1_id}\t{ids_str}\n")

print(f"  ✅ Saved: {matching_path}")

# ── candidate_pairs.tsv ──────────────────────────────────────
candidate_path = os.path.join(CFG.OUTPUT_DIR, "candidate_pairs.tsv")
with open(candidate_path, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1_id in s1_test["entity_id"].values:
        cands = candidate_results.get(s1_id, set())
        ids_str = ",".join(sorted(cands)) if cands else ""
        f.write(f"{s1_id}\t{ids_str}\n")

print(f"  ✅ Saved: {candidate_path}")

# %% ============================================================
# CELL 17 — VALIDATE SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: Validating submission files")
print("=" * 60)

# Use the provided validation script
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

# %% ============================================================
# CELL 18 — SAVE MODEL & CONFIG (for reproducibility)
# ============================================================
import json, pickle

# Save the trained model
model_path = os.path.join(CFG.OUTPUT_DIR, "final_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(final_model, f)
print(f"  ✅ Model saved: {model_path}")

# Save config & tuning results
config_out = {
    "best_threshold":     best_threshold,
    "val_f05":            val_f05,
    "best_params":        best_params,
    "feature_names":      FEATURE_NAMES,
    "blocking_recall":    block_recall,
    "n_train_pairs":      int(len(pairs_train)),
    "n_val_pairs":        int(len(pairs_val)),
}
config_path = os.path.join(CFG.OUTPUT_DIR, "run_config.json")
with open(config_path, "w") as f:
    json.dump(config_out, f, indent=2, default=str)
print(f"  ✅ Config saved: {config_path}")

# %% [markdown]
# ## Summary
#
# | Stage | Detail |
# |-------|--------|
# | **Blocking** | Country partitioning + TF-IDF char (3,5)-grams, top-50 |
# | **Features** | 18 features: name sims, address sims, source type |
# | **Model** | LightGBM (Optuna-tuned) |
# | **Evaluation** | Macro F_0.5 (precision-heavy) |
# | **Threshold** | Grid-searched on validation set |
#
# ### Next Steps to Improve Score
# - Increase `TRAIN_SAMPLE_S1` for more training data
# - Add phonetic features (Soundex / Metaphone) for name matching
# - Try multiple blocking passes (name-only + address-only) and union
# - Experiment with embedding-based blocking (sentence-transformers)
# - Add cross-field features (e.g., address tokens appearing in name)
# - Ensemble with XGBoost or CatBoost

print("\n" + "=" * 60)
print("🎉  PIPELINE COMPLETE!")
print("=" * 60)
print(f"  📁 Output directory: {CFG.OUTPUT_DIR}")
print(f"  📄 matching_results.tsv  — upload to leaderboard")
print(f"  📄 candidate_pairs.tsv   — include in submission zip")
print(f"  🎯 Threshold: {best_threshold:.2f}  |  Val F_0.5: {val_f05:.4f}")
