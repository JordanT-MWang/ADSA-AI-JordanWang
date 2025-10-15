import os
import argparse

def delete_bmp_files(root_folder):
    deleted_count = 0

    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for file in filenames:
            if (file.lower().endswith(".bmp") or file.lower().endswith(".tif") ):
                file_path = os.path.join(dirpath, file)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    
                except Exception as e:

                    print(f"⚠️ Could not delete")
        print(f"🗑️ Deleted: {filenames}")

    print(f"\n✅ Done! Deleted {deleted_count} .bmp files total.")


def main():
    parser = argparse.ArgumentParser(
        description="Deletes all .bmp files from a directory and its subdirectories."
    )
    parser.add_argument("folder_path", type=str, help="Path to the root folder to clean.")
    args = parser.parse_args()

    delete_bmp_files(args.folder_path)


if __name__ == "__main__":
    main()
