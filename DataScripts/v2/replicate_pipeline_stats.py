import json
import numpy as np
import argparse
import tensorflow as tf
import os
import sys

# import your custom generator
script_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
from DataGeneratorv3 import ADSADataPipeline


def inspect_generator(gen, max_batches=500):
    """
    Reads actual batches from the generator and captures the exact parameters
    that the generator outputs (after preprocessing, float32 casting, batching).
    """
    all_params = []
    print("Collecting batches...")

    for i, batch in enumerate(gen):
        if i >= max_batches:
            break

        if not isinstance(batch, tuple):
            raise ValueError(f"Unexpected batch format: {type(batch)}")

        inputs, labels = batch

        if not isinstance(inputs, tuple) or len(inputs) != 2:
            raise ValueError(f"Expected inputs as tuple(image, param), got: {type(inputs)}")

        images, params = inputs  # images ignored, params are what we want

        # Convert to numpy
        params_np = params.numpy()
        all_params.append(params_np)

    if len(all_params) == 0:
        raise RuntimeError("No batches collected!")

    all_params = np.concatenate(all_params, axis=0)
    print(f"Total collected samples: {all_params.shape}")

    return all_params


def compute_stats(params):
    """
    Compute float64 + float32 statistics.
    """
    params64 = params.astype(np.float64)
    params32 = params.astype(np.float32)

    return {
        "float64": {
            "mean": params64.mean(axis=0).tolist(),
            "std": params64.std(axis=0).tolist(),
            "min": params64.min(axis=0).tolist(),
            "max": params64.max(axis=0).tolist(),
            "unique_counts": [len(np.unique(params64[:, i])) for i in range(params64.shape[1])]
        },
        "float32": {
            "mean": params32.mean(axis=0).tolist(),
            "std": params32.std(axis=0).tolist(),
            "min": params32.min(axis=0).tolist(),
            "max": params32.max(axis=0).tolist(),
            "unique_counts": [len(np.unique(params32[:, i])) for i in range(params32.shape[1])]
        },
        "notes": {
            "total_samples": int(params.shape[0]),
            "n_parameters": int(params.shape[1])
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset root")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 640])
    parser.add_argument("--output", default="generator_stats.json")
    parser.add_argument("--max_batches", type=int, default=500)
    args = parser.parse_args()

    print("Initializing generator...")
    gen = ADSADataPipeline(
        args.dataset,
        split="train",
        image_size=tuple(args.image_size),
        output_type="Surface Tension (mN/m)",
        batch_size=args.batch_size
    ).get_dataset()

    params = inspect_generator(gen, max_batches=args.max_batches)
    stats = compute_stats(params)

    # Output path inside dataset folder
    out_path = os.path.join(args.dataset, args.output)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\nSaved statistics to {out_path}")
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    main()
