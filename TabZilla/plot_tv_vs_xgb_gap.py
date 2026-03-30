import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_selection_metric(target_type):
    if target_type == "regression":
        return "MSE", "min"
    if target_type == "binary":
        return "AUC", "max"
    if target_type == "classification":
        return "Log Loss", "min"
    raise ValueError(f"Unsupported target type: {target_type}")


def load_result_jsons(results_dir):
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
    if scorers is None:
        return False
    return "val" in scorers and "test" in scorers


def compute_dataset_performance(results_dir):
    trial_results = load_result_jsons(results_dir)
    valid_trials = [trial for trial in trial_results if is_valid_trial_result(trial)]
    if not valid_trials:
        raise RuntimeError(f"No valid trial results found in {results_dir}")

    reference = valid_trials[0]["data"]
    dataset_info = reference["dataset"]
    model_name = reference["model"]["name"]
    metric_name, direction = get_selection_metric(dataset_info["target_type"])

    reference_metric_values = reference["scorers"]["val"].get(metric_name)
    if reference_metric_values is None:
        raise RuntimeError(
            f"Validation metric '{metric_name}' not found in {results_dir}"
        )

    num_folds = len(reference_metric_values)
    fold_results = []

    for fold_idx in range(num_folds):
        best_trial = None
        best_val_value = None
        best_test_value = None
        best_test_accuracy = None

        for trial in valid_trials:
            val_values = trial["data"]["scorers"]["val"].get(metric_name)
            test_values = trial["data"]["scorers"]["test"].get(metric_name)
            if val_values is None or test_values is None:
                continue
            if fold_idx >= len(val_values) or fold_idx >= len(test_values):
                continue

            val_value = float(val_values[fold_idx])
            test_value = float(test_values[fold_idx])
            if np.isnan(val_value) or np.isnan(test_value):
                continue

            is_better = False
            if best_val_value is None:
                is_better = True
            elif direction == "min" and val_value < best_val_value:
                is_better = True
            elif direction == "max" and val_value > best_val_value:
                is_better = True

            if is_better:
                best_trial = trial
                best_val_value = val_value
                best_test_value = test_value
                accuracy_values = trial["data"]["scorers"]["test"].get("Accuracy")
                if accuracy_values is not None and fold_idx < len(accuracy_values):
                    accuracy_value = float(accuracy_values[fold_idx])
                    best_test_accuracy = None if np.isnan(accuracy_value) else accuracy_value
                else:
                    best_test_accuracy = None

        if best_trial is None:
            raise RuntimeError(
                f"Could not select best trial for fold {fold_idx} in {results_dir}"
            )

        fold_results.append(
            {
                "fold": fold_idx,
                "best_trial_name": best_trial["path"].name,
                "best_trial_number": best_trial["data"].get("trial_number"),
                "best_hparam_source": best_trial["data"].get("hparam_source"),
                "selection_metric_name": metric_name,
                "selection_metric_direction": direction,
                "selection_metric_value": best_val_value,
                "selected_test_value": best_test_value,
                "selected_test_accuracy": best_test_accuracy,
            }
        )

    selected_test_values = [fold_result["selected_test_value"] for fold_result in fold_results]
    selected_test_accuracies = [
        fold_result["selected_test_accuracy"]
        for fold_result in fold_results
        if fold_result["selected_test_accuracy"] is not None
    ]
    return {
        "dataset": dataset_info["name"],
        "target_type": dataset_info["target_type"],
        "model_name": model_name,
        "selection_metric_name": metric_name,
        "selection_metric_direction": direction,
        "mean_test_performance": float(np.mean(selected_test_values)),
        "std_test_performance": float(np.std(selected_test_values)),
        "mean_test_accuracy": (
            float(np.mean(selected_test_accuracies))
            if len(selected_test_accuracies) == num_folds
            else None
        ),
        "std_test_accuracy": (
            float(np.std(selected_test_accuracies))
            if len(selected_test_accuracies) == num_folds
            else None
        ),
        "num_valid_trials": len(valid_trials),
        "num_folds": num_folds,
        "fold_results": fold_results,
    }


def load_tv_summary_map(tv_summary_dir):
    aggregate_path = tv_summary_dir / "all_datasets_tv_summary.json"
    if aggregate_path.exists():
        with aggregate_path.open("r", encoding="utf-8") as file_handle:
            aggregate = json.load(file_handle)
        return {
            item["dataset"]: {
                "dataset": item["dataset"],
                "mean_tv": float(item["mean_tv"]),
                "std_tv": float(item["std_tv"]),
                "summary_path": item["summary_path"],
            }
            for item in aggregate["datasets"]
        }

    tv_map = {}
    for summary_path in sorted(tv_summary_dir.glob("*_tv_summary.json")):
        with summary_path.open("r", encoding="utf-8") as file_handle:
            summary = json.load(file_handle)
        tv_map[summary["dataset"]["name"]] = {
            "dataset": summary["dataset"]["name"],
            "mean_tv": float(summary["mean_tv"]),
            "std_tv": float(summary["std_tv"]),
            "summary_path": str(summary_path),
        }

    if not tv_map:
        raise FileNotFoundError(f"No TV summaries found in {tv_summary_dir}")
    return tv_map


def dataset_label(dataset_name):
    parts = dataset_name.split("__")
    if len(parts) >= 2:
        return parts[1]
    return dataset_name


def metric_gap(xgb_value, other_value, direction):
    if direction == "max":
        return xgb_value - other_value
    if direction == "min":
        return other_value - xgb_value
    raise ValueError(f"Unsupported direction: {direction}")


def metric_gap_label(metric_name, direction):
    if direction == "max":
        return f"{metric_name} gap (XGBoost - model)"
    return f"{metric_name} gap (model - XGBoost)"


def normalized_accuracy_gap(xgb_accuracy, model_accuracy):
    if xgb_accuracy is None or model_accuracy is None:
        return None
    if abs(model_accuracy) < 1e-12:
        return None
    return (xgb_accuracy - model_accuracy) / model_accuracy


def line_style_for_model(model_name):
    if model_name in {"XGBoost", "CatBoost", "LightGBM"}:
        return "--"
    return "-"


def discover_model_datasets(results_root, model_names):
    dataset_map = {}
    for model_name in model_names:
        model_dir = results_root / model_name
        dataset_names = set()
        if model_dir.exists():
            for dataset_dir in model_dir.iterdir():
                if dataset_dir.is_dir():
                    dataset_names.add(dataset_dir.name)
        dataset_map[model_name] = dataset_names
    return dataset_map


def build_gap_records(results_root, tv_map, model_names, require_complete_datasets):
    dataset_coverage = discover_model_datasets(results_root, model_names)
    tv_datasets = set(tv_map)

    if require_complete_datasets:
        included_datasets = set(tv_datasets)
        for dataset_names in dataset_coverage.values():
            included_datasets &= dataset_names
    else:
        included_datasets = set(tv_datasets) & dataset_coverage["XGBoost"]

    performance_summaries = {}
    skipped_model_datasets = []
    for model_name in model_names:
        for dataset_name in sorted(included_datasets):
            dataset_results_dir = results_root / model_name / dataset_name
            if not dataset_results_dir.exists():
                skipped_model_datasets.append(
                    {
                        "dataset": dataset_name,
                        "model_name": model_name,
                        "reason": "results_dir_missing",
                    }
                )
                continue

            try:
                performance_summaries[(model_name, dataset_name)] = compute_dataset_performance(
                    dataset_results_dir
                )
            except Exception as error:
                skipped_model_datasets.append(
                    {
                        "dataset": dataset_name,
                        "model_name": model_name,
                        "reason": str(error),
                    }
                )

    usable_datasets = set(included_datasets)
    for dataset_name in list(usable_datasets):
        for model_name in model_names:
            if (model_name, dataset_name) not in performance_summaries:
                usable_datasets.discard(dataset_name)
                break

    records = []
    competitor_models = [model_name for model_name in model_names if model_name != "XGBoost"]
    for dataset_name in sorted(usable_datasets):
        xgb_summary = performance_summaries[("XGBoost", dataset_name)]
        for model_name in competitor_models:
            model_summary = performance_summaries[(model_name, dataset_name)]
            if (
                xgb_summary["selection_metric_name"] != model_summary["selection_metric_name"]
                or xgb_summary["selection_metric_direction"] != model_summary["selection_metric_direction"]
            ):
                raise RuntimeError(
                    f"Metric mismatch for dataset {dataset_name}: "
                    f"XGBoost uses {xgb_summary['selection_metric_name']} / "
                    f"{xgb_summary['selection_metric_direction']}, "
                    f"{model_name} uses {model_summary['selection_metric_name']} / "
                    f"{model_summary['selection_metric_direction']}"
                )

            records.append(
                {
                    "dataset": dataset_name,
                    "dataset_label": dataset_label(dataset_name),
                    "target_type": xgb_summary["target_type"],
                    "metric_name": xgb_summary["selection_metric_name"],
                    "metric_direction": xgb_summary["selection_metric_direction"],
                    "model_name": model_name,
                    "tv_mean": tv_map[dataset_name]["mean_tv"],
                    "tv_std": tv_map[dataset_name]["std_tv"],
                    "xgb_test_performance": xgb_summary["mean_test_performance"],
                    "xgb_test_performance_std": xgb_summary["std_test_performance"],
                    "model_test_performance": model_summary["mean_test_performance"],
                    "model_test_performance_std": model_summary["std_test_performance"],
                    "xgb_test_accuracy": xgb_summary["mean_test_accuracy"],
                    "xgb_test_accuracy_std": xgb_summary["std_test_accuracy"],
                    "model_test_accuracy": model_summary["mean_test_accuracy"],
                    "model_test_accuracy_std": model_summary["std_test_accuracy"],
                    "xgb_advantage": metric_gap(
                        xgb_summary["mean_test_performance"],
                        model_summary["mean_test_performance"],
                        xgb_summary["selection_metric_direction"],
                    ),
                    "accuracy_gap": (
                        xgb_summary["mean_test_accuracy"] - model_summary["mean_test_accuracy"]
                        if xgb_summary["mean_test_accuracy"] is not None
                        and model_summary["mean_test_accuracy"] is not None
                        else None
                    ),
                    "normalized_accuracy_gap": normalized_accuracy_gap(
                        xgb_summary["mean_test_accuracy"],
                        model_summary["mean_test_accuracy"],
                    ),
                }
            )

    skipped_datasets = sorted(tv_datasets - usable_datasets)
    return {
        "records": records,
        "usable_datasets": sorted(usable_datasets),
        "initial_included_datasets": sorted(included_datasets),
        "skipped_datasets": skipped_datasets,
        "skipped_model_datasets": skipped_model_datasets,
        "performance_summaries": performance_summaries,
        "dataset_coverage": dataset_coverage,
    }


def save_records_csv(records, output_path):
    fieldnames = [
        "dataset",
        "dataset_label",
        "target_type",
        "metric_name",
        "metric_direction",
        "model_name",
        "tv_mean",
        "tv_std",
        "xgb_test_performance",
        "xgb_test_performance_std",
        "model_test_performance",
        "model_test_performance_std",
        "xgb_test_accuracy",
        "xgb_test_accuracy_std",
        "model_test_accuracy",
        "model_test_accuracy_std",
        "xgb_advantage",
        "accuracy_gap",
        "normalized_accuracy_gap",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def plot_gap_records(
    records,
    output_path,
    title,
    y_label,
    y_value_key,
    metric_legend_title="Metric",
):
    if not records:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    model_names = sorted({record["model_name"] for record in records})
    metric_names = sorted({record["metric_name"] for record in records})
    color_map = {
        model_name: plt.get_cmap("tab10")(idx % 10)
        for idx, model_name in enumerate(model_names)
    }
    marker_map = {
        metric_name: marker
        for metric_name, marker in zip(metric_names, ["o", "s", "^", "D", "P", "X"])
    }

    for record in records:
        ax.scatter(
            record["tv_mean"],
            record[y_value_key],
            color=color_map[record["model_name"]],
            marker=marker_map[record["metric_name"]],
            s=80,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.annotate(
            record["dataset_label"],
            (record["tv_mean"], record[y_value_key]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            alpha=0.8,
        )

    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[model_name],
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=8,
            linestyle="None",
            label=model_name,
        )
        for model_name in model_names
    ]
    metric_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker_map[metric_name],
            color="black",
            markersize=8,
            linestyle="None",
            label=metric_name,
        )
        for metric_name in metric_names
    ]

    legend1 = ax.legend(handles=model_handles, title="Model", loc="best")
    ax.add_artist(legend1)
    ax.legend(handles=metric_handles, title=metric_legend_title, loc="upper left")

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Mean TV distance")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_line_gap_records(
    records,
    output_path,
    title,
    y_label,
    y_value_key,
    metric_legend_title="Metric",
):
    if not records:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    model_names = sorted({record["model_name"] for record in records})
    metric_names = sorted({record["metric_name"] for record in records})
    color_map = {
        model_name: plt.get_cmap("tab10")(idx % 10)
        for idx, model_name in enumerate(model_names)
    }
    marker_map = {
        metric_name: marker
        for metric_name, marker in zip(metric_names, ["o", "s", "^", "D", "P", "X"])
    }

    records_by_model = defaultdict(list)
    for record in records:
        records_by_model[record["model_name"]].append(record)

    for model_name in model_names:
        model_records = sorted(
            records_by_model[model_name],
            key=lambda record: (record["tv_mean"], record["dataset"]),
        )
        ax.plot(
            [record["tv_mean"] for record in model_records],
            [record[y_value_key] for record in model_records],
            color=color_map[model_name],
            linewidth=1.8,
            alpha=0.8,
            linestyle=line_style_for_model(model_name),
        )

        for record in model_records:
            ax.scatter(
                record["tv_mean"],
                record[y_value_key],
                color=color_map[model_name],
                marker=marker_map[record["metric_name"]],
                s=80,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.5,
            )
            ax.annotate(
                record["dataset_label"],
                (record["tv_mean"], record[y_value_key]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                alpha=0.8,
            )

    model_handles = [
        plt.Line2D(
            [0],
            [0],
            color=color_map[model_name],
            linewidth=2.0,
            linestyle=line_style_for_model(model_name),
            label=model_name,
        )
        for model_name in model_names
    ]
    metric_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker_map[metric_name],
            color="black",
            markersize=8,
            linestyle="None",
            label=metric_name,
        )
        for metric_name in metric_names
    ]

    legend1 = ax.legend(handles=model_handles, title="Model", loc="best")
    ax.add_artist(legend1)
    ax.legend(handles=metric_handles, title=metric_legend_title, loc="upper left")

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Mean TV distance")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot XGBoost TV against task-aligned performance gaps."
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("TabZilla/results_hard/hard_run_main"),
        help="Root directory containing model result folders.",
    )
    parser.add_argument(
        "--tv_summary_dir",
        type=Path,
        default=None,
        help="Directory containing xgb_tv_posthoc summaries. Defaults to <results_root>/xgb_tv_posthoc.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for joined tables and plots. Defaults to <tv_summary_dir>/tv_vs_xgb_gap.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model list. XGBoost will be added automatically if omitted.",
    )
    parser.add_argument(
        "--include_partial_datasets",
        action="store_true",
        help="Allow datasets that are missing some competitor models. By default only complete datasets are used.",
    )
    args = parser.parse_args()

    tv_summary_dir = args.tv_summary_dir or (args.results_root / "xgb_tv_posthoc")
    output_dir = args.output_dir or (tv_summary_dir / "tv_vs_xgb_gap")
    output_dir.mkdir(parents=True, exist_ok=True)

    available_model_names = sorted(
        directory.name
        for directory in args.results_root.iterdir()
        if directory.is_dir() and directory.name != "xgb_tv_posthoc"
    )
    if "XGBoost" not in available_model_names:
        raise RuntimeError(f"XGBoost results not found in {args.results_root}")

    if args.models is None:
        model_names = available_model_names
    else:
        requested = set(args.models)
        requested.add("XGBoost")
        model_names = [model_name for model_name in available_model_names if model_name in requested]
        missing = sorted(requested - set(model_names))
        if missing:
            raise RuntimeError(f"Requested models not found: {missing}")

    tv_map = load_tv_summary_map(tv_summary_dir)
    build_output = build_gap_records(
        results_root=args.results_root,
        tv_map=tv_map,
        model_names=model_names,
        require_complete_datasets=not args.include_partial_datasets,
    )
    records = build_output["records"]
    if not records:
        raise RuntimeError("No usable dataset/model records found for plotting.")

    csv_path = output_dir / "tv_vs_xgb_gap_records.csv"
    save_records_csv(records, csv_path)

    performance_summary_payload = {
        "results_root": str(args.results_root.resolve()),
        "tv_summary_dir": str(tv_summary_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "model_names": model_names,
        "used_complete_dataset_intersection": not args.include_partial_datasets,
        "initial_included_datasets": build_output["initial_included_datasets"],
        "usable_datasets": build_output["usable_datasets"],
        "skipped_datasets": build_output["skipped_datasets"],
        "skipped_model_datasets": build_output["skipped_model_datasets"],
        "records": records,
        "performance_summaries": {
            f"{model_name}::{dataset_name}": summary
            for (model_name, dataset_name), summary in build_output["performance_summaries"].items()
        },
    }
    json_path = output_dir / "tv_vs_xgb_gap_summary.json"
    with json_path.open("w", encoding="utf-8") as file_handle:
        json.dump(performance_summary_payload, file_handle, indent=2)

    combined_line_plot_path = output_dir / "tv_vs_xgb_advantage_line.png"
    plot_line_gap_records(
        records=records,
        output_path=combined_line_plot_path,
        title="Mean TV vs XGBoost Advantage",
        y_label="Task-aligned gap (positive means XGBoost better)",
        y_value_key="xgb_advantage",
    )

    records_by_metric = defaultdict(list)
    for record in records:
        records_by_metric[record["metric_name"]].append(record)

    for metric_name, metric_records in sorted(records_by_metric.items()):
        metric_direction = metric_records[0]["metric_direction"]
        metric_line_plot_path = output_dir / (
            f"tv_vs_xgb_advantage_{metric_name.lower().replace(' ', '_')}_line.png"
        )
        plot_line_gap_records(
            records=metric_records,
            output_path=metric_line_plot_path,
            title=f"Mean TV vs XGBoost Advantage on {metric_name}",
            y_label=metric_gap_label(metric_name, metric_direction),
            y_value_key="xgb_advantage",
        )

    accuracy_records = [
        record for record in records if record["accuracy_gap"] is not None
    ]
    if accuracy_records:
        accuracy_records_by_target_type = defaultdict(list)
        for record in accuracy_records:
            accuracy_records_by_target_type[record["target_type"]].append(record)

    print(f"Wrote records CSV to: {csv_path}")
    print(f"Wrote summary JSON to: {json_path}")
    print(f"Wrote combined line plot to: {combined_line_plot_path}")
    if accuracy_records:
        for target_type, target_records in sorted(accuracy_records_by_target_type.items()):
            accuracy_line_plot_path = (
                output_dir / f"tv_vs_xgb_accuracy_gap_{target_type}_line.png"
            )
            plot_line_gap_records(
                records=target_records,
                output_path=accuracy_line_plot_path,
                title=f"Mean TV vs XGBoost Accuracy Gap ({target_type})",
                y_label="Accuracy gap (XGBoost - model)",
                y_value_key="accuracy_gap",
                metric_legend_title="Selection Metric",
            )
            print(f"Wrote accuracy-gap line plot to: {accuracy_line_plot_path}")

        normalized_accuracy_records = [
            record for record in accuracy_records if record["normalized_accuracy_gap"] is not None
        ]
        normalized_accuracy_records_by_target_type = defaultdict(list)
        for record in normalized_accuracy_records:
            normalized_accuracy_records_by_target_type[record["target_type"]].append(record)

        for target_type, target_records in sorted(normalized_accuracy_records_by_target_type.items()):
            normalized_accuracy_line_plot_path = (
                output_dir / f"tv_vs_xgb_normalized_accuracy_gap_{target_type}_line.png"
            )
            plot_line_gap_records(
                records=target_records,
                output_path=normalized_accuracy_line_plot_path,
                title=f"Mean TV vs XGBoost Normalized Accuracy Gap ({target_type})",
                y_label="Normalized accuracy gap ((XGBoost - model) / model)",
                y_value_key="normalized_accuracy_gap",
                metric_legend_title="Selection Metric",
            )
            print(
                f"Wrote normalized accuracy-gap line plot to: "
                f"{normalized_accuracy_line_plot_path}"
            )
    print(f"Usable datasets: {build_output['usable_datasets']}")


if __name__ == "__main__":
    main()
