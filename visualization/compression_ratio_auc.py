import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

# Parse input arguments
parser = argparse.ArgumentParser(description='Visualize compression ratio vs AUC.')
parser.add_argument('csv_file', type=str, help='Path to the CSV file containing the data.')
args = parser.parse_args()

# Load the data from the specified CSV file
data = pd.read_csv(args.csv_file)

# Filter out the 'full' method
full_data = data[data['Method'] == 'full']
other_data = data[data['Method'] != 'full']

# Plot the data
plt.figure(figsize=(10, 6))

# Plot the baseline 'full' method
if not full_data.empty:  # Check if full_data is not empty
    plt.axhline(y=full_data['Best AUC'].values[0], color='r', linestyle='--', label='full (baseline)')
else:
    print("Warning: full_data is empty. Cannot plot Best AUC.")

# Sort the other_data by 'Compression Rate' for each method before plotting
for method in other_data['Method'].unique():
    method_data = other_data[other_data['Method'] == method].sort_values('Compression Rate')
    plt.plot(method_data['Compression Rate'], method_data['Best AUC'], marker='o', label=method)

# Set plot labels and title
plt.xlabel('Compression Rate')
plt.ylabel('AUC')
plt.title('Compression Rate vs AUC for Different Methods')
plt.legend()
plt.grid(True)

# Create Figures directory if it doesn't exist
import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
figures_dir = os.path.join(current_dir, 'Figures')
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# Save the plot as a PDF file
plt.savefig(os.path.join(figures_dir, 'compression_ratio_auc.pdf'))
plt.show()



