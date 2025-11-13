import os
import argparse

def delete_inner_input_csv(base_dir, dry_run=False):
    deleted = []
    for root, dirs, files in os.walk(base_dir):
        if "Edges" in dirs:
            edges_path = os.path.join(root, "Edges")
            input_csv = os.path.join(edges_path, "input_params.csv")  # inside Edges folder
            if os.path.exists(input_csv):
                if dry_run:
                    print(f"[DRY RUN] Would delete: {input_csv}")
                else:
                    os.remove(input_csv)
                    print(f"Deleted: {input_csv}")
                deleted.append(input_csv)
    return deleted


def main():
    parser = argparse.ArgumentParser(description="Delete input_params.csv files inside Edges folders.")
    parser.add_argument("base_dir", type=str, help="Base directory to search for Edges folders.")
    parser.add_argument("--dry-run", action="store_true", help="Preview files to delete without actually deleting them.")
    args = parser.parse_args()

    deleted = delete_inner_input_csv(args.base_dir, dry_run=args.dry_run)
    print(f"\nTotal files {'to delete' if args.dry_run else 'deleted'}: {len(deleted)}")


if __name__ == "__main__":
    main()
