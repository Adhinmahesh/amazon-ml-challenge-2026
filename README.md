# Amazon ML Challenge 2026 — Business Entity Resolution

A complete end-to-end machine learning pipeline for resolving business entities across multiple data sources. Given business records from three different sources (Source 1, Source 2, Source 3), the system identifies which records refer to the same real-world business entity.

## Problem Statement

Match business entities across three data sources using name, address, and country information. Evaluation metric: **Macro F₀.₅** (precision-weighted).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PIPELINE OVERVIEW                     │
├─────────────────────────────────────────────────────────┤
│  Phase 1 │ Partition data by country to disk (streaming) │
│  Phase 2 │ Blocking + Feature extraction (per country)   │
│  Phase 3 │ Optuna hyperparameter tuning + LightGBM       │
│  Phase 4 │ Country-by-country test inference              │
│  Phase 5 │ Save output files & validate                   │
└─────────────────────────────────────────────────────────┘
```

## Key Features

- **Memory-Optimized**: Runs on machines with as little as 15 GB RAM using disk-based country partitioning
- **Checkpointing**: Stop and resume at any phase — progress is saved to disk automatically
- **18 Similarity Features**: TF-IDF cosine, Levenshtein, Jaro-Winkler, token sort/set ratio, Jaccard n-grams, and more
- **Automated Hyperparameter Tuning**: 25-trial Optuna search with F₀.₅ optimization
- **Sub-batch Processing**: Large country partitions (>2M records) are processed in smaller sub-batches

## 📁 Project Structure

```
haws/
├── solution_local.py          # Memory-optimized pipeline (15 GB RAM)
├── solution_colab.py          # Original pipeline (for Kaggle/Colab with 30+ GB RAM)
├── .env.example               # Environment configuration template
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── 6ab10eb3b23ba_student_resource/
│   └── student_resource/
│       ├── README.md          # Problem statement details
│       ├── dataset/
│       │   ├── train/         # Training source TSVs + ground truth
│       │   └── test/          # Test source TSVs
│       ├── utils/
│       │   └── validate_submission.py
│       ├── partitions/        # Auto-generated country partitions
│       ├── checkpoints/       # Auto-generated phase checkpoints
│       └── output/            # Generated submission files
│
└── venv/                      # Python virtual environment (not tracked)
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/haws.git
cd haws
```

### 2. Set Up Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Paths

```bash
cp .env.example .env
# Edit .env with your data paths if different
```

### 4. Download Dataset

Download the dataset from the Amazon ML Challenge portal and extract it into the project directory:
```
haws/6ab10eb3b23ba_student_resource/student_resource/dataset/
```

### 5. Run the Pipeline

**For local machines (15 GB RAM):**
```bash
python solution_local.py
```

**For Kaggle/Colab (30+ GB RAM):**
```bash
python solution_colab.py
```

## Checkpointing (Stop & Resume)

The local pipeline supports full checkpointing. You can stop at any time with `Ctrl+C` and resume later:

```bash
# First run — completes Phase 1 and Phase 2, then you stop
python solution_local.py
# Ctrl+C

# Second run — skips Phase 1 & 2, continues from Phase 3
python solution_local.py
```

To start completely fresh:
```bash
rm -rf 6ab10eb3b23ba_student_resource/student_resource/checkpoints
rm -rf 6ab10eb3b23ba_student_resource/student_resource/partitions
```

## 🔧 Configuration

Key parameters in `solution_local.py` (Config class):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOP_K` | 25 | Candidates per S1 entity during blocking |
| `TFIDF_MAX_FEATURES` | 50,000 | Max TF-IDF vocabulary size |
| `TRAIN_SAMPLE_FRAC` | 0.015 | Fraction of S1 entities sampled for training |
| `OPTUNA_TRIALS` | 25 | Number of hyperparameter search trials |
| `OPTUNA_TIMEOUT` | 1800s | Max time for tuning (30 min) |
| `FINAL_N_ESTIMATORS` | 2000 | Max boosting rounds for final model |

## Pipeline Details

### Blocking Strategy
- **Country partitioning**: Only compare entities within the same country
- **TF-IDF character n-grams** (3-4 grams): Build sparse similarity matrix
- **Sub-batch processing**: Countries with >2M S23 records are split into 2M-row batches

### Features (18 total)
| Category | Features |
|----------|----------|
| **Name** | TF-IDF cosine, Levenshtein, Jaro-Winkler, Token Sort Ratio, Token Set Ratio, Partial Ratio, Jaccard 3-grams, Jaccard words, Common words count, Length ratio |
| **Address** | Levenshtein, Jaro-Winkler, Token Sort Ratio, Jaccard words, Common words count, Length ratio, Either-empty flag |
| **Source** | is_source3 indicator |

### Model
- **LightGBM** binary classifier with Optuna-tuned hyperparameters
- **Threshold optimization**: Grid search on validation set for F₀.₅

## Output Files

After the pipeline completes:
- `matching_results.tsv` — Upload to the leaderboard
- `candidate_pairs.tsv` — Include in submission zip
- `final_model.pkl` — Trained model (for reproducibility)
- `run_config.json` — Run configuration and metrics

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas** — Data manipulation
- **scikit-learn** — TF-IDF vectorization
- **sparse-dot-topn** — Efficient sparse cosine similarity
- **rapidfuzz** — Fast string similarity (Levenshtein, Jaro-Winkler, fuzzy ratios)
- **LightGBM** — Gradient boosting classifier
- **Optuna** — Hyperparameter optimization
- **tqdm** — Progress bars

## License

This project was created for the Amazon ML Challenge 2026. All rights reserved.
