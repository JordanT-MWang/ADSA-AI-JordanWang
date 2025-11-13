import os
import csv
import cv2
import argparse

def read_params(params_path):
    """Read Scale Factor and Density from params.txt"""
    scale_factor = None
    delta_rho = None

    if not os.path.exists(params_path):
        print(f"⚠️ No params.txt found at {params_path}")
        return None, None

    with open(params_path, 'r') as f:
        for line in f:
            if "Scale Factor" in line:
                try:
                    scale_factor = float(line.split(":")[1].strip())
                except:
                    pass
            elif "Density" in line:
                try:
                    delta_rho = float(line.split(":")[1].split()[0].strip())
                except:
                    pass

    if scale_factor is None or delta_rho is None:
        print(f"⚠️ Could not read valid Scale Factor/Density from {params_path}")

    return delta_rho, scale_factor


def generate_csv(edges_folder, delta_rho, scale_factor):
    """Generate input_params.csv for all images in edges_folder"""
    if delta_rho is None or scale_factor is None:
        print(f"❌ Missing parameters for {edges_folder}, skipping.")
        return

    file_list = os.listdir(edges_folder)
    valid_extensions = {'.tif', '.tiff', '.bmp', '.png', '.jpg', '.jpeg'}

    # save CSV in the parent folder (where params.txt is)
    parent_folder = os.path.dirname(edges_folder)
    output_path = os.path.join(parent_folder, "input_params.csv")

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name', 'Delta Rho (g/ml)', 'Scale Factor (cm/pixel)', 'Resolution'])

        for name in file_list:
            _, ext = os.path.splitext(name)
            if ext.lower() not in valid_extensions:
                continue

            image_path = os.path.join(edges_folder, name)
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ Could not read image: {image_path}")
                continue

            height, width = img.shape[:2]
            resolution = f"{width} x {height}"

            writer.writerow([name, delta_rho, scale_factor, resolution])

    print(f"✅ CSV saved: {output_path}")


def process_directory(base_dir):
    """
    For each subdirectory in base_dir, go one layer deeper to find folders
    that contain an 'Edges' folder and a 'params.txt', then run generate_csv().
    """
    for root, dirs, files in os.walk(base_dir):
        if 'Edges' in dirs and 'params.txt' in files:
            params_path = os.path.join(root, 'params.txt')
            edges_path = os.path.join(root, 'Edges')

            delta_rho, scale_factor = read_params(params_path)
            if delta_rho and scale_factor:
                print(f"\nProcessing {edges_path}")
                print(f"  Δρ = {delta_rho}, Scale = {scale_factor}")
                generate_csv(edges_path, delta_rho, scale_factor)


def main():
    parser = argparse.ArgumentParser(
        description="Generate CSVs for ADSA AI dataset using params.txt values in each experiment folder."
    )
    parser.add_argument('-d', '--directory', required=True, type=str,
                        help="Path to the top-level directory containing experiment subfolders.")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"❌ Invalid directory path: {args.directory}")
        return

    process_directory(args.directory)


if __name__ == "__main__":
    main()
