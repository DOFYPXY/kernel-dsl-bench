#!/usr/bin/env python3
"""Visualize rmsnorm benchmark results as curves showing time vs hidden dimension."""

import argparse
import csv
import os
from collections import defaultdict


def parse_float(value: str):
	if value is None:
		return None
	text = value.strip()
	if text.upper() in {"N/A", "NA", ""}:
		return None
	try:
		return float(text)
	except ValueError:
		return None


def load_rmsnorm_results(csv_path: str):
	grouped = defaultdict(list)
	with open(csv_path, newline="") as csv_file:
		reader = csv.DictReader(csv_file)
		required = {
			"impl",
			"hidden",
			"mean_ms",
			"stddev_ms",
		}
		missing = required - set(reader.fieldnames or [])
		if missing:
			raise ValueError(f"CSV missing required columns: {sorted(missing)}")

		for row in reader:
			impl = (row.get("impl") or "").strip()
			hidden = parse_float(row.get("hidden", ""))
			mean_ms = parse_float(row.get("mean_ms", ""))
			std_ms = parse_float(row.get("stddev_ms", ""))

			if not impl or hidden is None or mean_ms is None:
				continue

			grouped[impl].append(
				{
					"hidden": int(hidden),
					"mean_ms": mean_ms,
					"std_ms": std_ms,
				}
			)

	# Sort each implementation's data by hidden dimension
	for impl in grouped:
		grouped[impl].sort(key=lambda x: x["hidden"])

	return grouped


def plot_size_curves(data: dict, output_dir: str, show: bool):
	try:
		import matplotlib.pyplot as plt
	except ImportError as error:
		raise RuntimeError(
			"matplotlib is required for plotting. Install with: pip install matplotlib"
		) from error

	if not data:
		raise ValueError("No data to plot")

	fig, ax = plt.subplots(figsize=(12, 4))

	for impl, rows in sorted(data.items()):
		if not rows:
			continue

		hidden_dims = [r["hidden"] for r in rows]
		means = [r["mean_ms"] for r in rows]
		stds = [0.0 if r["std_ms"] is None else r["std_ms"] for r in rows]

		ax.errorbar(
			hidden_dims,
			means,
			yerr=stds,
			marker="o",
			capsize=4,
			label=impl.upper(),
			linewidth=2,
			markersize=6,
		)

	ax.set_title("RMSNorm: Time vs Hidden Dimension")
	ax.set_xlabel("Hidden Dimension")
	ax.set_ylabel("Mean Time (ms)")
	ax.grid(True, linestyle="--", alpha=0.4)
	ax.legend()
	ax.set_ylim(bottom=0)

	os.makedirs(output_dir, exist_ok=True)
	output_path = os.path.join(output_dir, "rmsnorm_size_curves.png")
	fig.tight_layout()
	fig.savefig(output_path, dpi=160)
	print(f"Saved: {output_path}")

	if show:
		plt.show()
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser(
		description="Plot rmsnorm benchmark curves showing time vs hidden dimension"
	)
	parser.add_argument(
		"csv_path",
		nargs="?",
		help="Path to rmsnorm benchmark CSV (torch/triton/tilelang results)",
	)
	parser.add_argument(
		"--tk-csv",
		default=None,
		metavar="TK_CSV",
		help="Optional second CSV with TK results to overlay on the same plot",
	)
	parser.add_argument(
		"--output-dir",
		default="plots",
		help="Directory to save plots (default: plots)",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Display figure interactively",
	)
	args = parser.parse_args()

	if not os.path.exists(args.csv_path):
		raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

	data = load_rmsnorm_results(args.csv_path)

	if args.tk_csv:
		if not os.path.exists(args.tk_csv):
			raise FileNotFoundError(f"TK CSV file not found: {args.tk_csv}")
		tk_data = load_rmsnorm_results(args.tk_csv)
		data.update(tk_data)

	plot_size_curves(data, args.output_dir, args.show)


if __name__ == "__main__":
	main()
