#!/usr/bin/env bash

DATASETS=(
  "openml__albert__189356"
  "openml__artificial-characters__14964"
  "openml__audiology__7"
  "openml__balance-scale__11"
  "openml__cnae-9__9981"
  "openml__colic__25"
  "openml__credit-approval__29"
  "openml__credit-g__31"
  "openml__electricity__219"
  "openml__elevators__3711"
  "openml__guillermo__168337"
  "openml__heart-h__50"
  "openml__higgs__146606"
  "openml__jasmine__168911"
  "openml__jungle_chess_2pcs_raw_endgame_complete__167119"
  "openml__kc1__3917"
  "openml__lymph__10"
  "openml__mfeat-fourier__14"
  "openml__mfeat-zernike__22"
  "openml__monks-problems-2__146065"
  "openml__nomao__9977"
  "openml__one-hundred-plants-texture__9956"
  "openml__phoneme__9952"
  "openml__poker-hand__9890"
)

models=("LightGBM")
base_config="./tabzilla_experiment_config_gpu.yml"
python_bin="/home/ilia.kipshidze/projects/confounding-shift-estimation/Tabz/bin/python"
run_name="hard_run_LightGBM_batch_of_24"
failures=0

for ((i=${#DATASETS[@]}-1; i>=0; i--)); do
  ds="${DATASETS[i]}"
  dataset_dir="./datasets/$ds"

  if [ ! -d "$dataset_dir" ]; then
    echo "Skipping missing dataset: $ds"
    continue
  fi

  for model in "${models[@]}"; do
    out_dir="./results_hard/$run_name/$model/$ds"

    if [ -d "$out_dir" ] && [ "$(ls -A "$out_dir" 2>/dev/null)" ]; then
      echo "Skipping already completed model=$model dataset=$ds"
      continue
    fi

    mkdir -p "$out_dir"

    safe_model=$(echo "$model" | tr '/' '_' | tr ':' '_' | tr ' ' '_')
    safe_ds=$(echo "$ds" | tr '/' '_' | tr ':' '_' | tr ' ' '_')
    tmp_config="$(mktemp "./tmp_${safe_model}_${safe_ds}.XXXXXX.yml")"

    awk -v out="$out_dir" '
      BEGIN {done=0}
      /^output_dir:/ {print "output_dir: " out; done=1; next}
      {print}
      END {if (!done) print "output_dir: " out}
    ' "$base_config" > "$tmp_config"

    echo "Running model=$model dataset=$ds"

    if ! "$python_bin" tabzilla_experiment.py \
      --experiment_config "$tmp_config" \
      --dataset_dir "$dataset_dir" \
      --model_name "$model"; then
      echo "FAILED model=$model dataset=$ds"
      failures=$((failures + 1))
    fi

    rm -f "$tmp_config"
  done
done

echo "Finished with $failures failure(s)"
