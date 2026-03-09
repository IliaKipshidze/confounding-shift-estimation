import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_fold_points(results_dir: Path):
    points = []
    for result_file in sorted(results_dir.glob("*_results.json")):
        with result_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        tv_values = data.get("cluster_shift", {}).get("tv")
        mse_values = data.get("scorers", {}).get("test", {}).get("MSE")

        if not tv_values or not mse_values:
            continue

        if len(tv_values) != len(mse_values):
            raise ValueError(
                f"Mismatched fold counts in {result_file.name}: "
                f"{len(tv_values)} TV values vs {len(mse_values)} test MSE values"
            )

        label = data.get("hparam_source", result_file.stem)
        for fold_idx, (tv_value, mse_value) in enumerate(zip(tv_values, mse_values)):
            points.append(
                {
                    "label": label,
                    "tv": tv_value,
                    "test_mse": mse_value,
                    "file": result_file.name,
                    "fold": fold_idx,
                }
            )

    return points


def plot_tv_vs_test_mse(points, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))

    x = [point["tv"] for point in points]
    y = [point["test_mse"] for point in points]

    ax.scatter(x, y, alpha=0.8)

    for point in points:
        if point["label"] == "default":
            ax.annotate(
                point["label"],
                (point["tv"], point["test_mse"]),
                textcoords="offset points",
                xytext=(5, 5),
            )

    ax.set_xlabel("Total variation distance")
    ax.set_ylabel("Test MSE")
    ax.set_title("TV vs Test MSE Across Trials")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Plot foldwise cluster-shift TV against foldwise test MSE from result JSONs."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=script_dir / "results",
        help="Directory containing *_results.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "results" / "tv_vs_test_mse.png",
        help="Path to save the scatter plot.",
    )
    args = parser.parse_args()

    points = load_fold_points(args.results_dir)
    if not points:
        raise SystemExit(
            f"No result files with both cluster_shift.tv and scorers.test.MSE found in {args.results_dir}"
        )

    plot_tv_vs_test_mse(points, args.output)
    print(f"Loaded {len(points)} fold-level points.")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
