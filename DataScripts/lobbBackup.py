import os
import shutil
import argparse
#used to re-organize files from another system to be similar to our file system
def move_and_rename_dirs(parent_folder):
    moved_count = 0
    new_base = os.path.join(parent_folder, "MovedDirs")
    os.makedirs(new_base, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(parent_folder, topdown=False):
        # Skip the new base folder itself to avoid recursion
        if os.path.abspath(dirpath) == os.path.abspath(new_base):
            continue

        if "Edges" in dirnames and "data.dat" in filenames:
            original_name = os.path.basename(dirpath)
            new_name = f"{original_name}_{moved_count+1}"
            new_path = os.path.join(new_base, new_name)

            # Ensure unique name
            i = 1
            while os.path.exists(new_path):
                new_path = os.path.join(new_base, f"{new_name}_{i}")
                i += 1

            try:
                shutil.move(dirpath, new_path)
                moved_count += 1
                print(f"📂 Moved: {dirpath} → {new_path}")
            except Exception as e:
                print(f"⚠️ Failed to move {dirpath}: {e}")

    print(f"\n✅ Done! Moved {moved_count} directories into: {new_base}")


def main():
    parser = argparse.ArgumentParser(
        description="Move directories containing both 'data.dat' and 'Edges' into a uniquely named folder inside the parent directory."
    )
    parser.add_argument("folder_path", type=str, help="Path to the parent folder.")
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print("❌ Error: The provided path is not a valid directory.")
        return

    move_and_rename_dirs(args.folder_path)


if __name__ == "__main__":
    main()
