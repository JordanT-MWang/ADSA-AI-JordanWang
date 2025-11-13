import os
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import math

def combine_plots(input_dir, output_path, max_cols=3):
    """
    Combine all PNG plots in a directory into a single figure.

    Parameters:
        input_dir (str): Directory containing individual PNG distribution plots.
        output_path (str): Path to save the combined figure.
        max_cols (int): Maximum number of columns in the combined figure.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # List all PNG files
    png_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
    if not png_files:
        raise ValueError(f"No PNG files found in {input_dir}")

    png_files.sort()  # sort alphabetically

    n_files = len(png_files)
    n_cols = min(max_cols, n_files)
    n_rows = math.ceil(n_files / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_files > 1 else [axes]

    for ax, file in zip(axes, png_files):
        img_path = os.path.join(input_dir, file)
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(file.replace("distribution_", "").replace(".png", ""), fontsize=10)

    # Hide any extra axes
    for ax in axes[n_files:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Combined plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Combine all distribution plots into one figure.")
    parser.add_argument("input_dir", type=str, help="Directory containing distribution PNGs.")
    parser.add_argument("output_path", type=str, help="Path to save the combined figure.")
    parser.add_argument("--max_cols", type=int, default=3, help="Maximum number of columns in the figure.")
    args = parser.parse_args()

    combine_plots(args.input_dir, args.output_path, args.max_cols)


if __name__ == "__main__":
    main()
