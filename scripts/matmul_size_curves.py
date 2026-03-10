#!/usr/bin/env python3
"""Visualize matmul benchmark results as curves showing time vs matrix size."""

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


def load_matmul_results(csv_path: str):
	grouped = defaultdict(list)
	with open(csv_path, newline="") as csv_file:
		reader = csv.DictReader(csv_file)
		required = {
			"impl",
			"m",
			"mean_ms",
			"stddev_ms",
		}
		missing = required - set(reader.fieldnames or [])
		if missing:
			raise ValueError(f"CSV missing required columns: {sorted(missing)}")

		for row in reader:
			impl = (row.get("impl") or "").strip()
			size = parse_float(row.get("m", ""))
			mean_ms = parse_float(row.get("mean_ms", ""))
			std_ms = parse_float(row.get("stddev_ms", ""))

			if not impl or size is None or mean_ms is None:
				continue

			grouped[impl].append(
				{
					"size": int(size),
					"mean_ms": mean_ms,
					"std_ms": std_ms,
				}
			)

	# Sort each implementation's data by size
	for impl in grouped:
		grouped[impl].sort(key=lambda x: x["size"])

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

		sizes = [r["size"] for r in rows]
		means = [r["mean_ms"] for r in rows]
		stds = [0.0 if r["std_ms"] is None else r["std_ms"] for r in rows]

		ax.errorbar(
			sizes,
			means,
			yerr=stds,
			marker="o",
			capsize=4,
			label=impl.upper(),
			linewidth=2,
			markersize=6,
		)

	ax.set_title("MatMul: Time vs Matrix Size")
	ax.set_xlabel("Matrix Size (m=n=k)")
	ax.set_ylabel("Mean Time (ms)")
	ax.grid(True, linestyle="--", alpha=0.4)
	ax.legend()
	ax.set_ylim(bottom=0)

	os.makedirs(output_dir, exist_ok=True)
	output_path = os.path.join(output_dir, "matmul_size_curves.png")
	fig.tight_layout()
	fig.savefig(output_path, dpi=160)
	print(f"Saved: {output_path}")

	if show:
		plt.show()
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser(
		description="Plot matmul benchmark curves showing time vs matrix size"
	)
	parser.add_argument(
		"csv_path",
		nargs="?",
		help="Path to matmul benchmark CSV",
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

	data = load_matmul_results(args.csv_path)
	plot_size_curves(data, args.output_dir, args.show)


if __name__ == "__main__":
	main()
