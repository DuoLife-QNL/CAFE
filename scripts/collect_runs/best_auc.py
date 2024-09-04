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
    for filename in os.listdir(folder_path):
        if filename.endswith('.log'):
            file_path = os.path.join(folder_path, filename)
            best_auc = extract_best_auc(file_path)
            results.append([folder_path, filename, best_auc])
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Folder Path', 'Log File', 'Best AUC'])
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
