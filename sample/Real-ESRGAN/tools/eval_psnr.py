import cv2
import numpy as np
import os
from glob import glob
import argparse

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse))

def compare_folders(folder1, folder2):
    files1 = glob(os.path.join(folder1, '*'))
    files2 = glob(os.path.join(folder2, '*'))
    
    if len(files1) != len(files2):
        print("Warning: The number of files in the two folders is not the same!")
    
    psnr_values = []
    for f1, f2 in zip(sorted(files1), sorted(files2)):
        img1 = cv2.imread(f1)
        img2 = cv2.imread(f2)
        
        if img1 is None or img2 is None:
            print(f"Warning: Skipping comparison for {f1} and {f2} as one of them is not a valid image.")
            continue
        
        if img1.shape != img2.shape:
            print(f"Warning: The images {f1} and {f2} have different sizes and cannot be compared.")
            continue
        
        psnr = calculate_psnr(img1, img2)
        psnr_values.append(psnr)
        print(f"PSNR between {os.path.basename(f1)} and {os.path.basename(f2)}: {psnr}")
    
    if psnr_values:
        average_psnr = sum(psnr_values) / len(psnr_values)
        print("average_psnr: ", average_psnr)

def main():
    parser = argparse.ArgumentParser(description='Compare images in two folders.')
    parser.add_argument('--left_results', type=str, help='Path to the first folder')
    parser.add_argument('--right_results', type=str, help='Path to the second folder')
    
    args = parser.parse_args()
    
    compare_folders(args.left_results, args.right_results)

if __name__ == "__main__":
    main()
