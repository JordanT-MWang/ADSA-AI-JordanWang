import os
import shutil
import argparse

def combine_images(parent_folder):
    # Create a combined directory
    combined_dir = os.path.join(parent_folder, "combined_dir")
    os.makedirs(combined_dir, exist_ok=True)

    # Supported image extensions
    extensions = (".bmp", ".tif", ".tiff")

    count = 0
    for dirpath, _, filenames in os.walk(parent_folder):
        for file in filenames:
            if file.lower().endswith(extensions):
                src_path = os.path.join(dirpath, file)
                # Avoid overwriting by adding a counter if needed
                base_name, ext = os.path.splitext(file)
                dst_path = os.path.join(combined_dir, file)

                # If a file with the same name exists, append a number
                i = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(combined_dir, f"{base_name}_{i}{ext}")
                    i += 1

                shutil.copy2(src_path, dst_path)
                count += 1
                #print(f"📷 Copied: {src_path} → {dst_path}")

    print(f"\n✅ Done! {count} image files copied to: {combined_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine all .bmp, .tif, and .tiff images from subdirectories into one folder."
    )
    parser.add_argument("folder_path", type=str, help="Path to the parent folder.")
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print("❌ Error: The provided path is not a valid directory.")
        return

    combine_images(args.folder_path)


if __name__ == "__main__":
    main()
