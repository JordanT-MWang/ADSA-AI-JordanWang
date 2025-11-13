import os
import shutil
import argparse

def clean_and_check_directories(parent_folder):
    required_items = ["Edges", "data.dat", "params.txt"]
    missing_report = []

    for item in os.listdir(parent_folder):
        dir_path = os.path.join(parent_folder, item)
        if not os.path.isdir(dir_path):
            continue  # skip files in parent

        existing_items = os.listdir(dir_path)
        missing = [req for req in required_items if req not in existing_items]

        # Delete everything not in required_items
        for sub_item in existing_items:
            if sub_item not in required_items:
                sub_item_path = os.path.join(dir_path, sub_item)
                try:
                    if os.path.isdir(sub_item_path):
                        shutil.rmtree(sub_item_path)
                    else:
                        os.remove(sub_item_path)
                    print(f"🗑️ Deleted extra: {sub_item_path}")
                except Exception as e:
                    print(f"⚠️ Could not delete {sub_item_path}: {e}")

        # Record any missing required items
        if missing:
            missing_report.append((item, missing))

    # Print summary
    print("\n=== Summary Report ===")
    if not missing_report:
        print("✅ All directories have the required files and folders.")
    else:
        for folder, missing_items in missing_report:
            print(f"📁 {folder} → Missing: {', '.join(missing_items)}")
        print(f"\n⚠️ Total incomplete folders: {len(missing_report)}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean and check directories for 'Edges', 'data.dat', and 'params.txt'. Deletes other files."
    )
    parser.add_argument("folder_path", type=str, help="Path to the parent folder to check and clean.")
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print("❌ Error: The provided path is not a valid directory.")
        return

    clean_and_check_directories(args.folder_path)


if __name__ == "__main__":
    main()
