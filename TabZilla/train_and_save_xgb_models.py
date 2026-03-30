import argparse
import json
from pathlib import Path

from models.tree_models import XGBoost
from tabzilla_data_processing import process_data
from tabzilla_datasets import TabularDataset
from xgb_tv_pipeline_utils import (
    build_xgb_model_args,
    choose_best_trials_per_fold,
    load_result_jsons,
    sanitize_xgboost_params,
)


def train_and_save_dataset_models(
    dataset_dir,
    xgb_results_dataset_dir,
    output_dataset_dir,
    model_seed,
    force_cpu,
):
    dataset = TabularDataset.read(dataset_dir)
    trial_results = load_result_jsons(xgb_results_dataset_dir)
    best_trials = choose_best_trials_per_fold(trial_results, dataset.target_type)

    reference_experiment_args = best_trials[0]["trial_data"]["experiemnt_args"]
    dataset.subset_random_seed = reference_experiment_args.get("subset_random_seed", 0)

    model_args = build_xgb_model_args(
        dataset=dataset,
        experiment_args=reference_experiment_args,
        use_gpu=bool(reference_experiment_args.get("use_gpu", False) and not force_cpu),
    )

    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    fold_models = []

    for fold_idx, split_dictionary in enumerate(dataset.split_indeces):
        best_trial = best_trials[fold_idx]
        saved_params = best_trial["trial_data"]["model"]["params"]
        model_params = sanitize_xgboost_params(saved_params, seed=model_seed)

        processed = process_data(
            dataset,
            split_dictionary["train"],
            split_dictionary["val"],
            split_dictionary["test"],
            verbose=False,
            scaler=model_args.scale_numerical_features,
            one_hot_encode=False,
            args=model_args,
        )
        X_train, y_train = processed["data_train"]
        X_val, y_val = processed["data_val"]
        X_test, y_test = processed["data_test"]

        model = XGBoost(model_params.copy(), model_args)
        model.fit(X_train, y_train, X_val, y_val)

        model_filename = f"fold_{fold_idx}_xgboost_model.json"
        model_output_path = output_dataset_dir / model_filename
        model.model.save_model(model_output_path)

        fold_models.append(
            {
                "fold": fold_idx,
                "train_size": int(len(X_train)),
                "val_size": int(len(X_val)),
                "test_size": int(len(X_test)),
                "best_trial_name": best_trial["trial_name"],
                "best_trial_number": best_trial["trial_number"],
                "best_hparam_source": best_trial["hparam_source"],
                "selection_metric_name": best_trial["metric_name"],
                "selection_metric_direction": best_trial["metric_direction"],
                "selection_metric_value": best_trial["metric_value"],
                "selected_params": model_params,
                "model_filename": model_filename,
            }
        )

    manifest = {
        "dataset": dataset.get_metadata(),
        "dataset_dir": str(dataset_dir),
        "xgb_results_dataset_dir": str(xgb_results_dataset_dir),
        "saved_models_dataset_dir": str(output_dataset_dir),
        "reference_experiment_args": reference_experiment_args,
        "selection_metric": fold_models[0]["selection_metric_name"],
        "model_seed": int(model_seed),
        "force_cpu": bool(force_cpu),
        "fold_models": fold_models,
    }
    manifest_path = output_dataset_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle, indent=2)

    return manifest, manifest_path


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_xgb_results_dir = script_dir / "results_hard" / "hard_run_main" / "XGBoost"
    default_output_dir = script_dir / "results_hard" / "hard_run_main" / "xgb_saved_models"

    parser = argparse.ArgumentParser(
        description=(
            "Select the best saved XGBoost trial per fold by validation score, "
            "retrain that XGBoost once, and save the trained foldwise models."
        )
    )
    parser.add_argument(
        "--datasets_dir",
        type=Path,
        default=script_dir / "datasets",
        help="Directory containing the preprocessed dataset folders.",
    )
    parser.add_argument(
        "--xgb_results_dir",
        type=Path,
        default=default_xgb_results_dir,
        help="Directory containing per-dataset saved XGBoost result JSONs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_output_dir,
        help="Directory where trained foldwise XGBoost models and manifests will be written.",
    )
    parser.add_argument(
        "--model_seed",
        type=int,
        default=0,
        help="Seed used when retraining the saved XGBoost fold models.",
    )
    parser.add_argument(
        "--dataset_names",
        nargs="*",
        default=None,
        help="Optional subset of dataset folder names to process.",
    )
    parser.set_defaults(force_cpu=True)
    parser.add_argument(
        "--allow_gpu",
        dest="force_cpu",
        action="store_false",
        help="Use the saved GPU settings instead of forcing CPU retraining.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_result_dirs = [
        path
        for path in sorted(args.xgb_results_dir.iterdir())
        if path.is_dir() and (args.datasets_dir / path.name).is_dir()
    ]
    if args.dataset_names:
        allowed = set(args.dataset_names)
        dataset_result_dirs = [path for path in dataset_result_dirs if path.name in allowed]
    if not dataset_result_dirs:
        raise RuntimeError(
            f"No dataset result directories found in {args.xgb_results_dir} that match {args.datasets_dir}."
        )

    aggregate_manifest = {
        "xgb_results_dir": str(args.xgb_results_dir),
        "datasets_dir": str(args.datasets_dir),
        "output_dir": str(args.output_dir),
        "model_seed": int(args.model_seed),
        "force_cpu": bool(args.force_cpu),
        "datasets": [],
    }

    for result_dataset_dir in dataset_result_dirs:
        dataset_name = result_dataset_dir.name
        dataset_dir = args.datasets_dir / dataset_name
        output_dataset_dir = args.output_dir / dataset_name
        print(f"Training and saving fold models for {dataset_name}...")

        manifest, manifest_path = train_and_save_dataset_models(
            dataset_dir=dataset_dir,
            xgb_results_dataset_dir=result_dataset_dir,
            output_dataset_dir=output_dataset_dir,
            model_seed=args.model_seed,
            force_cpu=args.force_cpu,
        )

        aggregate_manifest["datasets"].append(
            {
                "dataset": dataset_name,
                "num_folds": len(manifest["fold_models"]),
                "manifest_path": str(manifest_path),
            }
        )

    aggregate_manifest_path = args.output_dir / "all_datasets_model_manifest.json"
    with aggregate_manifest_path.open("w", encoding="utf-8") as file_handle:
        json.dump(aggregate_manifest, file_handle, indent=2)

    print(f"Wrote aggregate manifest to {aggregate_manifest_path}")


if __name__ == "__main__":
    main()
