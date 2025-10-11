'''
This script generates the input parameter labels for all images
in a folder and saves it as a csv file

To run 
    python input.py /path/to/your/folder --delta_rho 0.99333 --scale_factor 0.00485437
    assuming that using ai generator adsa so images are in a folder above
    for my setup give the folder path to Edges file

'''

import os
import csv
import argparse

delta_rho = 0.995775
scale_factor = 0.000520492




source = r"H:\6-Amanda\Backup from Alligator (5-20-14)\Holder testing\5-3\b"
resolution = "1280 x 1024"
notes = ""

def generate_csv(folder_path):
    file_list = os.listdir(folder_path)

    valid_extensions = {'.tif', '.tiff', '.bmp', '.png'}

    # write csv file
    parent_folder = os.path.dirname(folder_path)
    output_path = os.path.join(parent_folder, "input_params.csv")
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # write headings
        writer.writerow(['Image Name', 'Delta Rho (g/ml)', 'Scale Factor (cm/pixel)', 'Resolution'])

        for name in file_list:
            # skip non image files
            _, ext = os.path.splitext(name)
            if ext.lower() not in valid_extensions:
                continue

            writer.writerow([name, delta_rho, scale_factor, resolution])





def main():
    parser = argparse.ArgumentParser(description="Generate a .csv file containing the input parameters matched to the images file name for all images in a folder.")
    parser.add_argument('folder_path', type=str, help="The path of the folder containing the image files.")
    parser.add_argument('--delta_rho', type=float, default=0.995775, help="Delta Rho value (g/ml)")
    parser.add_argument('--scale_factor', type=float, default=0.000520492, help="Scale Factor (cm/pixel)")
    args = parser.parse_args()
    
    global delta_rho, scale_factor
    delta_rho = args.delta_rho
    scale_factor = args.scale_factor
    generate_csv(args.folder_path)


if __name__ == "__main__":
    main()