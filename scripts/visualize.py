#!/usr/bin/env python3
"""Visualize benchmark CSV results as per-kernel scatter plots with error bars."""

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


def load_results(csv_path: str):
	grouped = defaultdict(list)
	with open(csv_path, newline="") as csv_file:
		reader = csv.DictReader(csv_file)
		required = {
			"Kernel",
			"Implementation",
			"Mean Time (ms)",
			"Stddev (ms)",
		}
		missing = required - set(reader.fieldnames or [])
		if missing:
			raise ValueError(f"CSV missing required columns: {sorted(missing)}")

		for row in reader:
			kernel = (row.get("Kernel") or "").strip()
			impl = (row.get("Implementation") or "").strip()
			mean_ms = parse_float(row.get("Mean Time (ms)", ""))
			std_ms = parse_float(row.get("Stddev (ms)", ""))

			if not kernel or not impl:
				continue

			grouped[kernel].append(
				{
					"impl": impl,
					"mean_ms": mean_ms,
					"std_ms": std_ms,
				}
			)
	return grouped


def plot_kernel(kernel: str, rows: list, output_dir: str, show: bool):
	try:
		import matplotlib.pyplot as plt
	except ImportError as error:
		raise RuntimeError(
			"matplotlib is required for plotting. Install with: pip install matplotlib"
		) from error

	valid_rows = [r for r in rows if r["mean_ms"] is not None]
	if not valid_rows:
		print(f"Skipping {kernel}: no numeric results to plot")
		return

	x_positions = list(range(len(valid_rows)))
	labels = [r["impl"] for r in valid_rows]
	means = [r["mean_ms"] for r in valid_rows]
	stds = [0.0 if r["std_ms"] is None else r["std_ms"] for r in valid_rows]

	fig, ax = plt.subplots(figsize=(4, 2.5))
	ax.errorbar(
		x_positions,
		means,
		yerr=stds,
		fmt="o",
		capsize=4,
		markersize=7,
		linewidth=1.2,
	)
	ax.set_title(f"{kernel.upper()}")
	ax.set_ylabel("Mean Time (ms)")
	ax.set_xticks(x_positions)
	ax.set_xticklabels(labels)
	ax.grid(axis="y", linestyle="--", alpha=0.4)
	ax.set_ylim(bottom=0) 

	os.makedirs(output_dir, exist_ok=True)
	output_path = os.path.join(output_dir, f"{kernel}_scatter.png")
	fig.tight_layout()
	fig.savefig(output_path, dpi=160)
	print(f"Saved: {output_path}")

	if show:
		plt.show()
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser(
		description="Read benchmark CSV and draw scatter plots with error bars per kernel"
	)
	parser.add_argument(
		"csv_path",
		nargs="?",
		help="Path to benchmark CSV",
	)
	parser.add_argument(
		"--output-dir",
		default="plots",
		help="Directory to save plots (default: plots)",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Display figures interactively",
	)
	args = parser.parse_args()

	if not os.path.exists(args.csv_path):
		raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

	grouped = load_results(args.csv_path)
	if not grouped:
		raise ValueError("No rows found in CSV")

	for kernel, rows in grouped.items():
		plot_kernel(kernel, rows, args.output_dir, args.show)


if __name__ == "__main__":
	main()
