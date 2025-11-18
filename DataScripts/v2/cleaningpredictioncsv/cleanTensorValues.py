import os
import argparse
import pandas as pd


def extract_num(x):
    x = str(x)
    if "tf.Tensor" in x:
        return float(x.split("(")[1].split(",")[0])
    return float(x)

def clean_csv(file_path):
    df = pd.read_csv(file_path)
    df["True_Value"] = df["True_Value"].apply(extract_num)
    df.to_csv(file_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    args = parser.parse_args()
    clean_csv(args.file_path)
if __name__=='__main__':
    main()