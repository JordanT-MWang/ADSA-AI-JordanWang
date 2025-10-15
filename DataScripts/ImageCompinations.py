import os
import shutil 
import argparse
def combine_bmp_images(parent_folder):
    #create new directory
    combined_dir = os.path.join(parent_folder,"combined_dir")
    os.makedirs(combined_dir, exist_ok=True)

    #get all subfolders
    subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f)) and f != "combined_dir"]

    #loop thorugh the sub folders 
    for subfolder in subfolders:
        folder__path = os.path.join(parent_folder, subfolder)
        bmp_files = [f for f in os.listdir(folder__path) if f.lower().endswith(".bmp")]

        for i, bmp_file in enumerate(bmp_files, start=1):
            #create unique names
            new_name = f"{subfolder}-{i}.bmp"
            src_path = os.path.join(folder__path,bmp_file)
            dst_path = os.path.join(combined_dir, new_name)

            #move the image to new directory
            shutil.move(src_path,dst_path)

        print(f"moved {folder__path} -> {dst_path}")

    print("moved all files!")

def main():
    parser = argparse.ArgumentParser(description="moves all imgages from sub folders into one combined folder with a unique names.")
    parser.add_argument('folder_path', type=str, help="The path of the folder containing the image folders.")

    args = parser.parse_args()

    combine_bmp_images(args.folder_path)


if __name__ == "__main__":
    main()