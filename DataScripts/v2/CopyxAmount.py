import os
import shutil
import argparse
from tqdm import tqdm

def sample_images(parent_dir, step=10, out_name="smaller_directory"):
    """
    Copies every `step`th image from parent_dir into a new folder.
    python sample_images.py "E:\Data\Jordan\Images" --step 10

    """

    # --- Make output directory ---
    out_dir = os.path.join(parent_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)

    # --- Allowed image extensions ---
    exts = {".png", ".jpg", ".jpeg", ".bmp"}

    # --- Collect all image names in sorted order ---
    images = sorted([
        f for f in os.listdir(parent_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])

    print(f"Found {len(images)} images.")
    print(f"Copying every {step}th image → {out_dir}")

    copied = 0

    for i in tqdm(range(0, len(images), step)):
        src = os.path.join(parent_dir, images[i])
        dst = os.path.join(out_dir, images[i])
        shutil.copy2(src, dst)
        copied += 1

    print(f"\n✨ Done! Copied {copied} images into {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sample every Nth image from a directory.")
    parser.add_argument("folder_path", type=str, help="Directory containing images")
    parser.add_argument("--step", type=int, default=10, help="Take every Nth image (default = 10)")
    parser.add_argument("--out_name", type=str, default="smaller_directory",
                        help="Name of output folder (default = smaller_directory)")

    args = parser.parse_args()
    sample_images(args.folder_path, args.step, args.out_name)


if __name__ == "__main__":
    main()
