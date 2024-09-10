import os
import re
import csv
import argparse

def extract_best_auc(file_path):
    best_auc = 0
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'auc (\d+\.\d+) %', line)
            if match:
                auc = float(match.group(1))
                best_auc = max(best_auc, auc)
    return best_auc

def process_folder(folder_path, output_csv):
    results = []
    for root, dirs, files in os.walk(folder_path):  # Recursively walk through the folder
        for filename in files:
            if filename.endswith('.log'):
                file_path = os.path.join(root, filename)  # Use root to get the correct file path
                best_auc = extract_best_auc(file_path)
                
                # Extract method and compression rate from the filename
                method, compression_rate = filename.rsplit('_', 1)  # Get everything before the last underscore as method
                compression_rate = compression_rate.replace('.log', '')  # Remove the .log extension
                
                results.append([root, filename, best_auc, method, compression_rate])  # Include method and compression rate
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Folder Path', 'Log File', 'Best AUC', 'Method', 'Compression Rate'])  # Added new headers
        writer.writerows(results)

def main():
    parser = argparse.ArgumentParser(description='Extract best AUC from log files.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing log files')
    parser.add_argument('--output', type=str, default='output.csv', help='Path to the output CSV file')
    args = parser.parse_args()

    process_folder(args.folder_path, args.output)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
