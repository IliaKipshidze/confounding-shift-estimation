import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder


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
        with result_path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
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


def build_xgb_model_args(dataset, experiment_args, use_gpu):
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
    )


def build_train_embedding(leaf_train, cluster_dim, cluster_seed):
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

    return z_train, reduction_method, int(reduction_dim)


def get_leaf_embeddings_from_booster(booster, X):
    dmatrix = xgb.DMatrix(X)
    leaf = booster.predict(dmatrix, pred_leaf=True)
    return leaf.astype("int32")


def predict_test_clusters_with_xgboost(
    X_train, c_train, X_test, cluster_seed, num_clusters
):
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
    tv = 0.5 * np.abs(p_train - p_test).sum()
    return float(tv), p_train.tolist(), p_test.tolist()


def cluster_train_embeddings(z_train, cluster_k, cluster_seed):
    used_k = min(cluster_k, len(z_train))
    kmeans = KMeans(
        n_clusters=used_k,
        random_state=cluster_seed,
        n_init="auto",
    )
    c_train = kmeans.fit_predict(z_train)
    return c_train, int(used_k)
