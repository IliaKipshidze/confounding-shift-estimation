import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xgboost as xgb
from models.tree_models import XGBoost
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from tabzilla_data_processing import process_data
from tabzilla_datasets import TabularDataset


def get_selection_metric(target_type):
    if target_type == "regression":
        return "MSE", "min"
    if target_type == "binary":
        return "AUC", "max"
    if target_type == "classification":
        return "Log Loss", "min"
    raise ValueError(f"Unsupported target type: {target_type}")


def load_result_jsons(results_dir: Path):
    trial_results = []
    for result_path in sorted(results_dir.glob("*_results.json")):
        with result_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        trial_results.append({"path": result_path, "data": data})
    if not trial_results:
        raise FileNotFoundError(f"No *_results.json files found in {results_dir}")
    return trial_results


def is_valid_trial_result(trial_result):
    data = trial_result["data"]
    if data.get("exception") not in (None, "None"):
        return False
    scorers = data.get("scorers")
    return scorers is not None and "val" in scorers


def choose_best_trials_per_fold(trial_results, target_type):
    metric_name, direction = get_selection_metric(target_type)
    valid_trials = [trial for trial in trial_results if is_valid_trial_result(trial)]
    if not valid_trials:
        raise RuntimeError("No valid trial results found.")

    first_metric_values = valid_trials[0]["data"]["scorers"]["val"][metric_name]
    num_folds = len(first_metric_values)
    best_trials = []

    for fold_idx in range(num_folds):
        best_trial = None
        best_value = None

        for trial in valid_trials:
            metric_values = trial["data"]["scorers"]["val"].get(metric_name)
            if metric_values is None or fold_idx >= len(metric_values):
                continue

            value = float(metric_values[fold_idx])
            if np.isnan(value):
                continue

            is_better = False
            if best_value is None:
                is_better = True
            elif direction == "min" and value < best_value:
                is_better = True
            elif direction == "max" and value > best_value:
                is_better = True

            if is_better:
                best_value = value
                best_trial = trial

        if best_trial is None:
            raise RuntimeError(
                f"Could not select a best trial for fold {fold_idx} using {metric_name}."
            )

        best_trials.append(
            {
                "fold": fold_idx,
                "metric_name": metric_name,
                "metric_direction": direction,
                "metric_value": best_value,
                "trial_path": str(best_trial["path"]),
                "trial_name": best_trial["path"].name,
                "trial_number": best_trial["data"].get("trial_number"),
                "hparam_source": best_trial["data"].get("hparam_source"),
                "trial_data": best_trial["data"],
            }
        )

    return best_trials


def sanitize_xgboost_params(saved_params, seed):
    excluded_keys = {
        "verbosity",
        "objective",
        "eval_metric",
        "tree_method",
        "gpu_id",
        "num_class",
    }
    cleaned = {
        key: value for key, value in saved_params.items() if key not in excluded_keys
    }
    cleaned["seed"] = seed
    return cleaned


def build_model_args(dataset, experiment_args, cluster_k, cluster_dim, cluster_seed, use_gpu):
    return SimpleNamespace(
        model_name="XGBoost",
        batch_size=experiment_args["batch_size"],
        scale_numerical_features=experiment_args["scale_numerical_features"],
        val_batch_size=experiment_args["val_batch_size"],
        objective=dataset.target_type,
        gpu_ids=experiment_args.get("gpu_ids", []),
        use_gpu=use_gpu,
        epochs=experiment_args["epochs"],
        data_parallel=bool(experiment_args.get("data_parallel", False) and use_gpu),
        early_stopping_rounds=experiment_args["early_stopping_rounds"],
        dataset=dataset.name,
        cat_idx=dataset.cat_idx,
        num_features=dataset.num_features,
        subset_features=experiment_args["subset_features"],
        subset_rows=experiment_args["subset_rows"],
        subset_features_method=experiment_args["subset_features_method"],
        subset_rows_method=experiment_args["subset_rows_method"],
        cat_dims=dataset.cat_dims,
        num_classes=dataset.num_classes,
        logging_period=experiment_args["logging_period"],
        compute_cluster_shift=False,
        cluster_k=cluster_k,
        cluster_pca_dim=cluster_dim,
        cluster_seed=cluster_seed,
    )


def build_train_embedding(leaf_train, cluster_dim, cluster_seed, leaf_encoding):
    if leaf_encoding == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoded_train = encoder.fit_transform(leaf_train)

        if encoded_train.shape[1] > 1 and cluster_dim > 0:
            reduction_dim = min(cluster_dim, encoded_train.shape[1] - 1)
            reducer = TruncatedSVD(
                n_components=reduction_dim,
                random_state=cluster_seed,
            )
            z_train = reducer.fit_transform(encoded_train)
            reduction_method = "truncated_svd"
        else:
            reduction_dim = encoded_train.shape[1]
            z_train = encoded_train.toarray()
            reduction_method = "none"
        return (
            encoder,
            reducer if reduction_method == "truncated_svd" else None,
            z_train,
            reduction_method,
            int(reduction_dim),
        )

    if leaf_encoding == "raw":
        raw_train = np.asarray(leaf_train, dtype=np.float32)

        if raw_train.shape[1] > 1 and cluster_dim > 0:
            reduction_dim = min(cluster_dim, raw_train.shape[1] - 1)
            reducer = TruncatedSVD(
                n_components=reduction_dim,
                random_state=cluster_seed,
            )
            z_train = reducer.fit_transform(raw_train)
            reduction_method = "truncated_svd"
        else:
            reduction_dim = raw_train.shape[1]
            z_train = raw_train
            reduction_method = "none"
        return (
            None,
            reducer if reduction_method == "truncated_svd" else None,
            z_train,
            reduction_method,
            int(reduction_dim),
        )

    raise ValueError(f"Unsupported leaf encoding: {leaf_encoding}")


def transform_embedding(leaf_data, leaf_encoding, encoder, reducer):
    if leaf_encoding == "onehot":
        transformed = encoder.transform(leaf_data)
        if reducer is not None:
            return reducer.transform(transformed)
        return transformed.toarray()

    if leaf_encoding == "raw":
        transformed = np.asarray(leaf_data, dtype=np.float32)
        if reducer is not None:
            return reducer.transform(transformed)
        return transformed

    raise ValueError(f"Unsupported leaf encoding: {leaf_encoding}")


def predict_test_clusters_with_xgboost(X_train, c_train, X_test, cluster_seed, num_clusters):
    if num_clusters == 1:
        return np.zeros(X_test.shape[0], dtype=int)

    classifier_params = {
        "random_state": cluster_seed,
        "verbosity": 0,
        "tree_method": "hist",
    }

    if num_clusters > 2:
        classifier_params["objective"] = "multi:softprob"
        classifier_params["num_class"] = num_clusters
        classifier_params["eval_metric"] = "mlogloss"
    else:
        classifier_params["objective"] = "binary:logistic"
        classifier_params["eval_metric"] = "logloss"

    classifier = xgb.XGBClassifier(**classifier_params)
    classifier.fit(X_train, c_train)
    return classifier.predict(X_test).astype(int)


def compute_tv_from_labels(c_train, c_test, num_clusters):
    p_train = np.bincount(c_train, minlength=num_clusters) / len(c_train)
    p_test = np.bincount(c_test, minlength=num_clusters) / len(c_test)
    tv = (1.0 / num_clusters) * np.abs(p_train - p_test).sum()
    return float(tv), p_train.tolist(), p_test.tolist()


def compute_dataset_tv(
    dataset_dir,
    xgb_results_dataset_dir,
    cluster_k,
    cluster_dim,
    cluster_seed,
    force_cpu,
    cluster_prediction_method,
    leaf_encoding,
):
    dataset = TabularDataset.read(dataset_dir)
    trial_results = load_result_jsons(xgb_results_dataset_dir)
    best_trials = choose_best_trials_per_fold(trial_results, dataset.target_type)

    reference_experiment_args = best_trials[0]["trial_data"]["experiemnt_args"]
    dataset.subset_random_seed = reference_experiment_args.get("subset_random_seed", 0)

    model_args = build_model_args(
        dataset=dataset,
        experiment_args=reference_experiment_args,
        cluster_k=cluster_k,
        cluster_dim=cluster_dim,
        cluster_seed=cluster_seed,
        use_gpu=bool(reference_experiment_args.get("use_gpu", False) and not force_cpu),
    )

    fold_summaries = []
    for fold_idx, split_dictionary in enumerate(dataset.split_indeces):
        best_trial = best_trials[fold_idx]
        saved_params = best_trial["trial_data"]["model"]["params"]
        model_params = sanitize_xgboost_params(saved_params, seed=cluster_seed)

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

        main_model = XGBoost(model_params.copy(), model_args)
        main_model.fit(X_train, y_train, X_val, y_val)

        leaf_train = main_model.get_leaf_embeddings(X_train)
        encoder, reducer, z_train, reduction_method, used_dim = build_train_embedding(
            leaf_train=leaf_train,
            cluster_dim=cluster_dim,
            cluster_seed=cluster_seed,
            leaf_encoding=leaf_encoding,
        )

        used_k = min(cluster_k, len(z_train))
        kmeans = KMeans(
            n_clusters=used_k,
            random_state=cluster_seed,
            n_init="auto",
        )
        c_train = kmeans.fit_predict(z_train)
        if cluster_prediction_method == "kmeans":
            leaf_test = main_model.get_leaf_embeddings(X_test)
            z_test = transform_embedding(
                leaf_data=leaf_test,
                leaf_encoding=leaf_encoding,
                encoder=encoder,
                reducer=reducer,
            )
            c_test = kmeans.predict(z_test)
        elif cluster_prediction_method == "xgb":
            c_test = predict_test_clusters_with_xgboost(
                X_train=X_train,
                c_train=c_train,
                X_test=X_test,
                cluster_seed=cluster_seed,
                num_clusters=used_k,
            )
        else:
            raise ValueError(
                f"Unsupported cluster prediction method: {cluster_prediction_method}"
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
                "best_trial_name": best_trial["trial_name"],
                "best_trial_number": best_trial["trial_number"],
                "best_hparam_source": best_trial["hparam_source"],
                "selection_metric_name": best_trial["metric_name"],
                "selection_metric_direction": best_trial["metric_direction"],
                "selection_metric_value": best_trial["metric_value"],
                "selected_params": model_params,
                "cluster_k_requested": int(cluster_k),
                "cluster_k_used": int(used_k),
                "cluster_dim_requested": int(cluster_dim),
                "cluster_dim_used": int(used_dim),
                "leaf_encoding": leaf_encoding,
                "embedding_reduction": reduction_method,
                "cluster_prediction_method": cluster_prediction_method,
                "tv": tv,
                "p_train": p_train,
                "p_test": p_test,
            }
        )

    tv_values = [fold["tv"] for fold in fold_summaries]
    return {
        "dataset": dataset.get_metadata(),
        "xgb_results_dir": str(xgb_results_dataset_dir),
        "selection_metric": fold_summaries[0]["selection_metric_name"],
        "cluster_k_requested": int(cluster_k),
        "cluster_dim_requested": int(cluster_dim),
        "cluster_seed": int(cluster_seed),
        "leaf_encoding": leaf_encoding,
        "cluster_prediction_method": cluster_prediction_method,
        "force_cpu": bool(force_cpu),
        "fold_results": fold_summaries,
        "mean_tv": float(np.mean(tv_values)),
        "std_tv": float(np.std(tv_values)),
    }


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_xgb_results_dir = (
        script_dir / "results_hard" / "hard_run_main" / "XGBoost"
    )
    default_output_dir = (
        script_dir / "results_hard" / "hard_run_main" / "xgb_tv_posthoc"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Select the best saved XGBoost trial per fold by validation score, retrain it, "
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
        "--xgb_results_dir",
        type=Path,
        default=default_xgb_results_dir,
        help="Directory containing per-dataset saved XGBoost result JSONs.",
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
        help="Random seed for retraining, dimensionality reduction, clustering, and cluster prediction.",
    )
    parser.add_argument(
        "--cluster_prediction_method",
        choices=("xgb", "kmeans"),
        default="xgb",
        help=(
            "How to assign test samples to train-defined clusters: "
            "'xgb' keeps the current auxiliary XGBoost classifier, "
            "'kmeans' uses direct k-means prediction in embedding space."
        ),
    )
    parser.add_argument(
        "--leaf_encoding",
        choices=("onehot", "raw"),
        default="onehot",
        help=(
            "How to represent XGBoost leaf embeddings before optional reduction and clustering: "
            "'onehot' uses categorical one-hot encoding, 'raw' uses raw leaf IDs directly."
        ),
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

    dataset_dirs = [
        path
        for path in sorted(args.xgb_results_dir.iterdir())
        if path.is_dir() and (args.datasets_dir / path.name).is_dir()
    ]
    if args.dataset_names:
        allowed = set(args.dataset_names)
        dataset_dirs = [path for path in dataset_dirs if path.name in allowed]
    if not dataset_dirs:
        raise RuntimeError(
            f"No dataset result directories found in {args.xgb_results_dir} that match {args.datasets_dir}."
        )

    aggregate_summary = {
        "xgb_results_dir": str(args.xgb_results_dir),
        "datasets_dir": str(args.datasets_dir),
        "output_dir": str(args.output_dir),
        "cluster_k": int(args.cluster_k),
        "cluster_dim": int(args.cluster_dim),
        "cluster_seed": int(args.cluster_seed),
        "leaf_encoding": args.leaf_encoding,
        "cluster_prediction_method": args.cluster_prediction_method,
        "force_cpu": bool(args.force_cpu),
        "datasets": [],
    }

    for result_dataset_dir in dataset_dirs:
        dataset_name = result_dataset_dir.name
        dataset_dir = args.datasets_dir / dataset_name
        print(f"Processing {dataset_name}...")

        dataset_summary = compute_dataset_tv(
            dataset_dir=dataset_dir,
            xgb_results_dataset_dir=result_dataset_dir,
            cluster_k=args.cluster_k,
            cluster_dim=args.cluster_dim,
            cluster_seed=args.cluster_seed,
            force_cpu=args.force_cpu,
            cluster_prediction_method=args.cluster_prediction_method,
            leaf_encoding=args.leaf_encoding,
        )

        output_path = args.output_dir / f"{dataset_name}_tv_summary.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(dataset_summary, f, indent=2)

        aggregate_summary["datasets"].append(
            {
                "dataset": dataset_name,
                "mean_tv": dataset_summary["mean_tv"],
                "std_tv": dataset_summary["std_tv"],
                "summary_path": str(output_path),
            }
        )
        print(f"  mean TV = {dataset_summary['mean_tv']:.6f}")

    aggregate_path = args.output_dir / "all_datasets_tv_summary.json"
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, indent=2)

    print(f"Wrote summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
