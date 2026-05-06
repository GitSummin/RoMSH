@echo off
setlocal EnableExtensions EnableDelayedExpansion

pushd "%~dp0"

if not defined PYTHON_BIN set "PYTHON_BIN=python"
if not defined DATA_DIR set "DATA_DIR=data"
if not defined RESULT_ROOT set "RESULT_ROOT=results_chemseq"
if not defined SUMMARY_CSV set "SUMMARY_CSV=%RESULT_ROOT%\proposed_all6_summary.csv"
if not exist "%RESULT_ROOT%" mkdir "%RESULT_ROOT%"

echo ============================================================
echo [Proposed Model - ALL 6 DATASETS]
echo PYTHON_BIN=%PYTHON_BIN%
echo DATA_DIR=%DATA_DIR%
echo RESULT_ROOT=%RESULT_ROOT%
echo SUMMARY_CSV=%SUMMARY_CSV%
echo ============================================================

rem ---------------------------------------------------------------
rem  %%D : split type folder   (e.g. scaffold, no_scaffold)
rem  %%C : case/split subfolder (e.g. example, split_1, ...)
rem  Add your own split names below to run on multiple datasets.
rem ---------------------------------------------------------------
for %%D in (no_scaffold scaffold) do (
  for %%C in (example) do (
    call :run_case "%%D" "%%C"
    if errorlevel 1 (
      echo [ERROR] Failed at %%D / %%C
      popd
      exit /b 1
    )
  )
)

echo.
echo [DONE] Proposed model finished for all 6 datasets.
echo Summary CSV: %SUMMARY_CSV%
popd
exit /b 0


:run_case
set "DATASET_KIND=%~1"
set "CASE_NAME=%~2"

set "CASE_DIR=%DATA_DIR%\%DATASET_KIND%\%CASE_NAME%"
set "TRAIN_CSV=%CASE_DIR%\train.csv"
set "VAL_CSV=%CASE_DIR%\val.csv"
set "TEST_CSV=%CASE_DIR%\test.csv"
set "CORPUS_CSV=%CASE_DIR%\%CASE_NAME%_smiles_corpus.csv"
if not exist "%CORPUS_CSV%" set "CORPUS_CSV=%CASE_DIR%\smiles_corpus.csv"

if not exist "%TRAIN_CSV%" (
  echo [SKIP] %DATASET_KIND% / %CASE_NAME% ^(train.csv not found^)
  exit /b 0
)
if not exist "%VAL_CSV%" (
  echo [SKIP] %DATASET_KIND% / %CASE_NAME% ^(val.csv not found^)
  exit /b 0
)
if not exist "%TEST_CSV%" (
  echo [SKIP] %DATASET_KIND% / %CASE_NAME% ^(test.csv not found^)
  exit /b 0
)
if not exist "%CORPUS_CSV%" (
  echo [ERROR] corpus csv not found for %DATASET_KIND% / %CASE_NAME%
  echo         checked: %CASE_DIR%\%CASE_NAME%_smiles_corpus.csv
  echo         checked: %CASE_DIR%\smiles_corpus.csv
  exit /b 1
)

set "OUT_DIR=%RESULT_ROOT%\all6_proposed\%DATASET_KIND%\%CASE_NAME%"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

set "MLM_ROOT=results_mlm\%DATASET_KIND%\%CASE_NAME%_mlm_result"
set "MLM_RUN_DIR="
if exist "%MLM_ROOT%" call :resolve_mlm_run_dir "%MLM_ROOT%"

if not defined MLM_RUN_DIR (
  echo [INFO] MLM checkpoint not found for %DATASET_KIND% / %CASE_NAME%. Pretraining MLM first...
  %PYTHON_BIN% pretrain_mlm.py ^
    --corpus_csv "%CORPUS_CSV%" ^
    --out_dir "%MLM_ROOT%" ^
    --max_seq_len 128 ^
    --max_tokens 1200 ^
    --mask_strategy span ^
    --span_len 3 ^
    --mask_prob 0.15 ^
    --d_model 256 ^
    --n_layers 4 ^
    --n_heads 4 ^
    --dropout 0.1 ^
    --epochs 30 ^
    --batch_size 256 ^
    --eval_batch_size 512 ^
    --lr 3e-4 ^
    --weight_decay 1e-2 ^
    --warmup_ratio 0.05 ^
    --min_lr_ratio 0.10 ^
    --amp
  if errorlevel 1 exit /b 1
  call :resolve_mlm_run_dir "%MLM_ROOT%"
)

if not defined MLM_RUN_DIR (
  echo [ERROR] Could not locate MLM checkpoint under %MLM_ROOT%
  exit /b 1
)

set "MLM_ENCODER=%MLM_RUN_DIR%\mlm_encoder.pt"
set "VOCAB_JSON=%MLM_RUN_DIR%\vocab.json"

echo ------------------------------------------------------------
echo [CASE] %DATASET_KIND% / %CASE_NAME%
echo TRAIN=%TRAIN_CSV%
echo VAL=%VAL_CSV%
echo TEST=%TEST_CSV%
echo CORPUS=%CORPUS_CSV%
echo MLM_RUN_DIR=%MLM_RUN_DIR%
echo OUT=%OUT_DIR%
echo ------------------------------------------------------------

%PYTHON_BIN% train_chemseq_proposed.py ^
  --train_csv "%TRAIN_CSV%" ^
  --val_csv "%VAL_CSV%" ^
  --test_csv "%TEST_CSV%" ^
  --dataset_tag %DATASET_KIND% ^
  --split_name %CASE_NAME% ^
  --rt_col rts ^
  --abundance_col abundance ^
  --fuel_col fuel_proxy ^
  --out_dir "%OUT_DIR%" ^
  --summary_csv "%SUMMARY_CSV%" ^
  --ablation_group proposed_all6 ^
  --experiment_name ProposedTheoryV2 ^
  --seed 123 ^
  --pretrained_mlm_encoder "%MLM_ENCODER%" ^
  --pretrained_vocab "%VOCAB_JSON%" ^
  --max_seq_len 128 ^
  --max_mols 5 ^
  --d_model 256 ^
  --token_layers 4 ^
  --token_heads 4 ^
  --token_dropout 0.1 ^
  --mix_layers 2 ^
  --mix_heads 4 ^
  --mix_dropout 0.15 ^
  --rt_fourier_K 8 ^
  --rt_norm minmax ^
  --use_pos_emb 1 ^
  --head_hidden 160 ^
  --head_dropout 0.25 ^
  --triplet_layers 0 ^
  --triplet_heads 4 ^
  --triplet_dropout 0.10 ^
  --grl_lambda 1.0 ^
  --use_hybrid_stats ^
  --epochs 60 ^
  --batch_size 64 ^
  --eval_batch_size 256 ^
  --lr 8e-4 ^
  --weight_decay 1e-2 ^
  --grad_clip 1.0 ^
  --early_stop 14 ^
  --stop_metric jaccard ^
  --token_mask_p_train 0.08 ^
  --token_mask_p_eval 0.0 ^
  --mol_drop_p 0.08 ^
  --mol_drop_min_keep 2 ^
  --thr_mode tune_on_val ^
  --tune_thr_metric jaccard ^
  --thr_strategy quantile ^
  --thr_grid 199 ^
  --temp_scale_on_val ^
  --event_loss focal ^
  --event_focal_gamma 1.5 ^
  --event_label_smoothing 0.01 ^
  --logit_l2_weight 1e-4 ^
  --neg_penalty_weight 0.06 ^
  --neg_penalty_power 2.0 ^
  --use_fuel_aux ^
  --fuel_loss_weight 0.15 ^
  --use_contrastive ^
  --contrastive_weight 0.05 ^
  --contrastive_temperature 0.07 ^
  --contrastive_hard_neg_k 8 ^
  --contrastive_min_pos 1 ^
  --use_context_aux ^
  --context_loss_weight 0.15 ^
  --use_context_adv ^
  --context_adv_weight 0.08 

if errorlevel 1 exit /b 1

echo [DONE] %DATASET_KIND% / %CASE_NAME%
exit /b 0


:resolve_mlm_run_dir
set "SEARCH_ROOT=%~1"
set "MLM_RUN_DIR="
for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$root = [System.IO.Path]::GetFullPath('%SEARCH_ROOT%'); if ((Test-Path (Join-Path $root 'mlm_encoder.pt')) -and (Test-Path (Join-Path $root 'vocab.json'))) { Write-Output $root; exit 0 }; $cand = Get-ChildItem -Path $root -Directory -Filter 'run_*' -ErrorAction SilentlyContinue ^| Sort-Object LastWriteTime -Descending ^| Select-Object -First 1; if ($cand -and (Test-Path (Join-Path $cand.FullName 'mlm_encoder.pt')) -and (Test-Path (Join-Path $cand.FullName 'vocab.json'))) { Write-Output $cand.FullName; exit 0 }; $enc = Get-ChildItem -Path $root -Recurse -File -Filter 'mlm_encoder.pt' -ErrorAction SilentlyContinue ^| Select-Object -First 1; if ($enc) { $dir = $enc.DirectoryName; if (Test-Path (Join-Path $dir 'vocab.json')) { Write-Output $dir; exit 0 } }"`) do set "MLM_RUN_DIR=%%I"
goto :eof