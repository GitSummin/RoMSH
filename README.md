# ChemSequence: Chemical Mixture Hazard Prediction via Sequence Modeling

**ChemSequence** is a deep learning framework for predicting fire hazard labels of multi-component chemical mixtures using GC-MS chromatographic data (SMILES sequences, retention times, and abundance profiles).

## Overview

Given a chemical mixture represented as a set of SMILES strings with retention times and abundance values, the model predicts whether the mixture is fire-hazardous. The proposed model (RoMSH) integrates:

- **MLM Pretraining** — Span-masked language modeling on SMILES corpora for molecule-level representation
- **TokenTransformerFP+** — Per-molecule SMILES encoder combining CLS, mean, max, and last-token pooling with attention-based fusion
- **Fourier RT Features** — Retention time encoding via Fourier positional features (K=8)
- **MixtureTransformerCLS** — Mixture-level transformer that models inter-molecular interactions
- **Hybrid Statistics Fusion** — Fuses CLS, importance-weighted, mean, and max aggregations via a learned gate
- **Multi-task Auxiliary Losses** — Focal loss, supervised contrastive learning, fuel-proxy prediction, and context adversarial training

## Repository Structure

```
ChemSequence/
├── run.bat          # Main training script for all 6 dataset splits
├── train.py      # Proposed model training entry point
├── pretrain_mlm.py                # SMILES MLM pretraining
├── metrics.py                     # Evaluation metrics (AUROC, AUPRC, Jaccard, etc.)
│
├── models/
│   ├── chemseq_model.py           # ChemSeqModel (main proposed architecture)
│   ├── loss_function.py           # ChemSeqLoss (focal, contrastive, fuel, context)
│   ├── mlm_model.py               # SmilesMLMModel for pretraining
│   ├── attn_transformer.py        # Pre-norm transformer blocks
│   ├── smiles_trfm_encoder.py     # SMILES token encoder with sinusoidal PE
│   └── triplet_interaction.py     # Optional higher-order triplet interaction layer
│
├── data/
│   ├── mixture_dataset.py         # MixtureCSVDataset (PyTorch Dataset)
│   ├── smiles_tokenizer.py        # SMILES tokenizer and vocabulary utilities
│   ├── smiles_mlm_dataset.py      # SmilesMLMDataset for pretraining
│   ├── scaffold/
│   │   └── example/               # Example scaffold-split dataset
│   │       ├── train.csv
│   │       ├── val.csv
│   │       ├── test.csv
│   │       └── smiles_corpus.csv
│   └── no_scaffold/
│       └── example/               # Example random-split dataset
│           ├── train.csv
│           ├── val.csv
│           ├── test.csv
│           └── smiles_corpus.csv
│
└── requirements.txt
```

> **Note:** The `data/` directory contains only small example files (10 training samples, 5 val/test samples, 20 SMILES corpus entries per split). The full dataset used in the paper is not included due to its size. To run training on your own data, place CSV files under `data/scaffold/<split_name>/` and `data/no_scaffold/<split_name>/`, then add the split name to the loop in `run_proposed_all6.bat`.

## Data Format

### `train.csv` / `val.csv` / `test.csv`

| Column | Description |
|--------|-------------|
| `mixture` | Space-separated SMILES strings (one per component) |
| `label` | Binary fire hazard label (1 = hazardous, 0 = non-hazardous) |
| `rts` | Space-separated retention times (in minutes, one per component) |
| `abundance` | Space-separated abundance fractions (sum to 1) |
| `fuel_proxy` | Continuous fuel score in [0, 1] (used as auxiliary supervision target) |
| `env_id` | Integer environment/domain ID (for context adversarial training) |
| `env_name` | Human-readable environment name |
| `gid` | Unique sample identifier |

### `*_smiles_corpus.csv`

| Column | Description |
|--------|-------------|
| `smiles` | One SMILES string per row (used for MLM pretraining) |

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
```

Install with:
```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Prepare your data

Place your datasets under `data/scaffold/` and `data/no_scaffold/`, following the CSV format described above. Each split subfolder (e.g. `split_1`, `split_2`) should contain:

```
data/scaffold/split_1/train.csv
data/scaffold/split_1/val.csv
data/scaffold/split_1/test.csv
data/scaffold/split_1/smiles_corpus.csv
```

Then add the split name to the loop in `run_proposed_all6.bat`:

```bat
for %%D in (no_scaffold scaffold) do (
  for %%C in (split_1 split_2 ...) do (
```

### Step 2: Run training

```bat
run_proposed_all6.bat
```

This script:
1. **Pretrains** a SMILES MLM encoder on the provided corpus (if not already cached under `results_mlm/`)
2. **Trains** the proposed ChemSeqModel on each split listed in the loop
3. **Evaluates** on the test set with threshold tuning and temperature scaling
4. **Aggregates** results into `results_chemseq/proposed_all6_summary.csv`

### Step 3: Custom runs

Run MLM pretraining manually:
```bash
python pretrain_mlm.py \
    --corpus_csv data/Scaffold/C1_random_id/C1_random_id_smiles_corpus.csv \
    --out_dir results_mlm/Scaffold/C1_random_id_mlm_result \
    --d_model 256 --n_layers 4 --n_heads 4 \
    --epochs 30 --batch_size 256 --amp
```

Run proposed model training:
```bash
python train_chemseq_proposed.py \
    --train_csv data/Scaffold/C1_random_id/train.csv \
    --val_csv data/Scaffold/C1_random_id/val.csv \
    --test_csv data/Scaffold/C1_random_id/test.csv \
    --pretrained_mlm_encoder results_mlm/Scaffold/C1_random_id_mlm_result/mlm_encoder.pt \
    --pretrained_vocab results_mlm/Scaffold/C1_random_id_mlm_result/vocab.json \
    --out_dir results_chemseq/run_example \
    --d_model 256 --token_layers 4 --mix_layers 2 \
    --epochs 60 --batch_size 64 --lr 8e-4 \
    --event_loss focal --use_fuel_aux --use_contrastive \
    --use_context_aux --use_context_adv \
    --thr_mode tune_on_val --temp_scale_on_val
```

## Model Architecture

```
Input: mixture = {(SMILES_i, RT_i, abundance_i)}

For each molecule i:
  SMILES_i → SmilesTrfmEncoder → [CLS, mean, max, last] → TokenTransformerFP+
  RT_i → FourierRTFeatures (K=8 Fourier bands)
  abundance_i → log-abundance embedding

Per-molecule representations are fused (Wm + Wr + Wa) and passed to:
  MixtureTransformerCLS → z_cls, h_mol[1..N]

Hybrid statistics fusion:
  [z_cls, z_imp, z_mean, z_max] → stat_proj → fuse_gate → z_mix

Classification head:
  z_mix → proj_chem, proj_ctx → event_feat → head → logit

Auxiliary outputs:
  fuel_head(z_chem) → fuel_proxy prediction
  context_head(z_ctx) → environment classification
  adv_context_head(GRL(z_chem)) → adversarial domain confusion
```

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| `d_model` | 256 |
| `token_layers` | 4 |
| `mix_layers` | 2 |
| `rt_fourier_K` | 8 |
| `epochs` | 60 |
| `batch_size` | 64 |
| `lr` | 8e-4 |
| `early_stop` | 14 epochs |
| `event_loss` | Focal (γ=1.5) |
| `fuel_loss_weight` | 0.15 |
| `contrastive_weight` | 0.05 |
| `context_loss_weight` | 0.15 |
| `context_adv_weight` | 0.08 |
| `thr_mode` | tune on validation (Jaccard) |
| `temp_scale_on_val` | ✓ |

## License

This project is for research purposes. Please cite the associated paper if you use this code in your work.
