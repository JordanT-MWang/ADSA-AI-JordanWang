import os
from PIL import Image
import argparse

def convert_png_to_bmp(parent_dir):
    """
    Convert all PNGs in the Edges/ folder to BMP and save them
    in a new folder edges_bmp/ next to the Edges/ folder.
    """
    edges_dir = os.path.join(parent_dir, "Edges")
    if not os.path.exists(edges_dir):
        raise FileNotFoundError(f"No Edges folder found in {parent_dir}")

    bmp_dir = os.path.join(parent_dir, "edges_bmp")
    os.makedirs(bmp_dir, exist_ok=True)

    png_files = [f for f in os.listdir(edges_dir) if f.lower().endswith(".png")]
    if not png_files:
        print(f"No PNG files found in {edges_dir}")
        return

    print(f"Converting {len(png_files)} PNG files to BMP...")

    for file_name in png_files:
        png_path = os.path.join(edges_dir, file_name)
        bmp_name = os.path.splitext(file_name)[0] + ".bmp"
        bmp_path = os.path.join(bmp_dir, bmp_name)

        # Open PNG and save as BMP
        with Image.open(png_path) as img:
            img.save(bmp_path, format="BMP")

        #print(f"Saved: {bmp_path}")

    print(f"\n✅ All PNGs converted to BMP in folder: {bmp_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert all PNGs in Edges/ to BMP.")
    parser.add_argument("parent_dir", type=str, help="Parent directory containing Edges folder.")
    args = parser.parse_args()

    convert_png_to_bmp(args.parent_dir)


if __name__ == "__main__":
    main()
