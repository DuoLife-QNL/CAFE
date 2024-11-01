import pickle
from typing import List, Any, Tuple, Dict
from Sample import Sample
from collections import defaultdict
import torch
import json
import os

def process_dataset(
    samples_path: str = 'datasets/moments/processed/samples_parsed.pkl',
    labels_path: str = 'datasets/moments/processed/labels.pkl',
    keys_info_path: str = 'datasets/moments/processed/keys_info.json',
    output_dir: str = 'datasets/moments/processed/'
) -> Tuple[List[Dict[str, Dict]], List[Any], Dict[str, Dict[int, int]], Dict[str, List[str]]]:
    """
    Load the parsed samples and labels from pickle files, process them, and save the results.

    Args:
        samples_path (str): Path to the pickle file containing the parsed samples.
        labels_path (str): Path to the pickle file containing the labels.
        keys_info_path (str): Path to the pickle file containing the keys information.
        output_dir (str): Directory to save the processed files.

    Returns:
        Tuple[List[Dict[str, Dict]], List[Any], Dict[str, Dict[int, int]], Dict[str, List[str]]]: A tuple containing the list of processed samples, 
        the list of labels, the global feature map, and the keys information.
    """
    def load_pickle(file_path: str) -> List[Any]:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Successfully loaded {len(data)} items from {file_path}")
            return data
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            raise
        except pickle.UnpicklingError:
            print(f"Error: There was a problem unpickling the file {file_path}.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while loading {file_path}: {str(e)}")
            raise

    samples_parsed, labels = load_pickle(samples_path), load_pickle(labels_path)

    # Create a global mapping dictionary for each field
    global_feature_maps = defaultdict(lambda: defaultdict(int))
    
    # Function to recursively process nested dictionaries
    def process_features(features, field=None):
        if isinstance(features, dict):
            if field is None:
                return {k: process_features(v, f"{k}") for k, v in features.items()}
            else:
                return {k: process_features(v, f"{field}.{k}") for k, v in features.items()}
        elif isinstance(features, list):
            return [process_features(item, field) for item in features]
        elif isinstance(features, torch.Tensor):
            return [process_feature(field, int(hash_id)) for hash_id in features.tolist()]
        else:
            return process_feature(field, int(features))

    def process_feature(field, feature):
        if feature not in global_feature_maps[field]:
            global_feature_maps[field][feature] = len(global_feature_maps[field])
        return global_feature_maps[field][feature]

    # Process all samples
    samples_processed = []
    for sample in samples_parsed:
        processed_sample = {
            "dense_dict": sample["dense_dict"],
            "sparse_dict": process_features(sample["sparse_dict"])
        }
        samples_processed.append(processed_sample)

    # Save processed samples
    processed_samples_path = os.path.join(output_dir, 'samples_processed.pkl')
    with open(processed_samples_path, 'wb') as f:
        pickle.dump(samples_processed, f)
    print(f"Saved processed samples to {processed_samples_path}")

    # Save global feature maps
    feature_map_path = os.path.join(output_dir, 'global_feature_maps.pkl')
    with open(feature_map_path, 'wb') as f:
        pickle.dump({field: dict(feature_map) for field, feature_map in global_feature_maps.items()}, f)
    print(f"Saved global feature maps to {feature_map_path}")

    # Calculate feature counts for each field
    feature_counts = {field: len(feature_map) for field, feature_map in global_feature_maps.items()}

    # Load keys information
    with open(keys_info_path, 'r') as f:
        keys_info = json.load(f)

    # Create metadata
    metadata = {
        "total_unique_features": sum(feature_counts.values()),
        "num_samples": len(samples_parsed),
        "feature_counts_per_field": feature_counts,
        "keys_info": keys_info
    }

    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    return samples_processed, labels, {field: dict(feature_map) for field, feature_map in global_feature_maps.items()}, keys_info

# Example usage:
samples, labels, feature_maps, keys_info = process_dataset()

print(f"Total unique features: {sum(len(fm) for fm in feature_maps.values())}")
print(f"Sample of feature maps:")
for field, fm in list(feature_maps.items())[:5]:
    print(f"  {field}: {dict(list(fm.items())[:5])}")
print(f"First processed sample: {samples[0]}")
print(f"Keys info: {keys_info}")
