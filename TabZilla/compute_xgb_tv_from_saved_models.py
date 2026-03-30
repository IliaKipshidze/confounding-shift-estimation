import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from tabzilla_data_processing import process_data
from tabzilla_datasets import TabularDataset
from xgb_tv_pipeline_utils import (
    build_train_embedding,
    build_xgb_model_args,
    cluster_train_embeddings,
    compute_tv_from_labels,
    get_leaf_embeddings_from_booster,
    predict_test_clusters_with_xgboost,
)


def compute_dataset_tv_from_saved_models(
    dataset_dir,
    saved_models_dataset_dir,
    cluster_k,
    cluster_dim,
    cluster_seed,
):
    manifest_path = saved_models_dataset_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file_handle:
        manifest = json.load(file_handle)

    dataset = TabularDataset.read(dataset_dir)
    reference_experiment_args = manifest["reference_experiment_args"]
    dataset.subset_random_seed = reference_experiment_args.get("subset_random_seed", 0)

    model_args = build_xgb_model_args(
        dataset=dataset,
        experiment_args=reference_experiment_args,
        use_gpu=False,
    )

    fold_summaries = []
    for fold_model in manifest["fold_models"]:
        fold_idx = fold_model["fold"]
        split_dictionary = dataset.split_indeces[fold_idx]

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

        booster = xgb.Booster()
        booster.load_model(saved_models_dataset_dir / fold_model["model_filename"])

        leaf_train = get_leaf_embeddings_from_booster(booster, X_train)
        z_train, reduction_method, used_dim = build_train_embedding(
            leaf_train=leaf_train,
            cluster_dim=cluster_dim,
            cluster_seed=cluster_seed,
        )

        c_train, used_k = cluster_train_embeddings(
            z_train=z_train,
            cluster_k=cluster_k,
            cluster_seed=cluster_seed,
        )
        c_test = predict_test_clusters_with_xgboost(
            X_train=X_train,
            c_train=c_train,
            X_test=X_test,
            cluster_seed=cluster_seed,
            num_clusters=used_k,
        )

        tv, p_train, p_test = compute_tv_from_labels(
            c_train=c_train,
            c_test=c_test,
            num_clusters=used_k,
        )

        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_size": int(len(X_train)),
                "val_size": int(len(X_val)),
                "test_size": int(len(X_test)),
                "best_trial_name": fold_model["best_trial_name"],
                "best_trial_number": fold_model["best_trial_number"],
                "best_hparam_source": fold_model["best_hparam_source"],
                "selection_metric_name": fold_model["selection_metric_name"],
                "selection_metric_direction": fold_model["selection_metric_direction"],
                "selection_metric_value": fold_model["selection_metric_value"],
                "selected_params": fold_model["selected_params"],
                "saved_model_filename": fold_model["model_filename"],
                "cluster_k_requested": int(cluster_k),
                "cluster_k_used": int(used_k),
                "cluster_dim_requested": int(cluster_dim),
                "cluster_dim_used": int(used_dim),
                "embedding_reduction": reduction_method,
                "tv": tv,
                "p_train": p_train,
                "p_test": p_test,
            }
        )

    tv_values = [fold["tv"] for fold in fold_summaries]
    return {
        "dataset": dataset.get_metadata(),
        "dataset_dir": str(dataset_dir),
        "saved_models_dataset_dir": str(saved_models_dataset_dir),
        "saved_models_manifest_path": str(manifest_path),
        "selection_metric": fold_summaries[0]["selection_metric_name"],
        "cluster_k_requested": int(cluster_k),
        "cluster_dim_requested": int(cluster_dim),
        "cluster_seed": int(cluster_seed),
        "model_seed": int(manifest["model_seed"]),
        "force_cpu_when_training_models": bool(manifest["force_cpu"]),
        "fold_results": fold_summaries,
        "mean_tv": float(np.mean(tv_values)),
        "std_tv": float(np.std(tv_values)),
    }


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_saved_models_dir = script_dir / "results_hard" / "hard_run_main" / "xgb_saved_models"
    default_output_dir = (
        script_dir / "results_hard" / "hard_run_main" / "xgb_tv_from_saved_models"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Load pre-saved foldwise XGBoost models, extract leaf embeddings, "
            "and compute foldwise / mean TV distances."
        )
    )
    parser.add_argument(
        "--datasets_dir",
        type=Path,
        default=script_dir / "datasets",
        help="Directory containing the preprocessed dataset folders.",
    )
    parser.add_argument(
        "--saved_models_dir",
        type=Path,
        default=default_saved_models_dir,
        help="Directory containing saved foldwise XGBoost models and manifests.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_output_dir,
        help="Directory where TV summaries will be written.",
    )
    parser.add_argument(
        "--cluster_k",
        type=int,
        default=20,
        help="Number of clusters to fit on train embeddings.",
    )
    parser.add_argument(
        "--cluster_dim",
        type=int,
        default=20,
        help="Dimension used after one-hot leaf encoding via TruncatedSVD.",
    )
    parser.add_argument(
        "--cluster_seed",
        type=int,
        default=0,
        help="Random seed for dimensionality reduction, clustering, and cluster prediction.",
    )
    parser.add_argument(
        "--dataset_names",
        nargs="*",
        default=None,
        help="Optional subset of dataset folder names to process.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    saved_dataset_dirs = [
        path
        for path in sorted(args.saved_models_dir.iterdir())
        if path.is_dir()
        and (path / "manifest.json").is_file()
        and (args.datasets_dir / path.name).is_dir()
    ]
    if args.dataset_names:
        allowed = set(args.dataset_names)
        saved_dataset_dirs = [path for path in saved_dataset_dirs if path.name in allowed]
    if not saved_dataset_dirs:
        raise RuntimeError(
            f"No saved model dataset directories found in {args.saved_models_dir} that match {args.datasets_dir}."
        )

    aggregate_summary = {
        "saved_models_dir": str(args.saved_models_dir),
        "datasets_dir": str(args.datasets_dir),
        "output_dir": str(args.output_dir),
        "cluster_k": int(args.cluster_k),
        "cluster_dim": int(args.cluster_dim),
        "cluster_seed": int(args.cluster_seed),
        "datasets": [],
    }

    for saved_dataset_dir in saved_dataset_dirs:
        dataset_name = saved_dataset_dir.name
        dataset_dir = args.datasets_dir / dataset_name
        print(f"Computing TV from saved models for {dataset_name}...")

        dataset_summary = compute_dataset_tv_from_saved_models(
            dataset_dir=dataset_dir,
            saved_models_dataset_dir=saved_dataset_dir,
            cluster_k=args.cluster_k,
            cluster_dim=args.cluster_dim,
            cluster_seed=args.cluster_seed,
        )

        output_path = args.output_dir / f"{dataset_name}_tv_summary.json"
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(dataset_summary, file_handle, indent=2)

        aggregate_summary["datasets"].append(
            {
                "dataset": dataset_name,
                "mean_tv": dataset_summary["mean_tv"],
                "std_tv": dataset_summary["std_tv"],
                "summary_path": str(output_path),
            }
        )

    aggregate_summary_path = args.output_dir / "all_datasets_tv_summary.json"
    with aggregate_summary_path.open("w", encoding="utf-8") as file_handle:
        json.dump(aggregate_summary, file_handle, indent=2)

    print(f"Wrote aggregate summary to {aggregate_summary_path}")


if __name__ == "__main__":
    main()
