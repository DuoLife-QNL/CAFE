import pickle
import json
from typing import List, Any, Tuple, Dict

def load_processed_dataset(
    samples_path: str = 'datasets/moments/processed/samples_processed.pkl',
    labels_path: str = 'datasets/moments/processed/labels.pkl',
    metadata_path: str = 'datasets/moments/processed/metadata.json',
) -> Tuple[List[Dict[str, Dict]], List[Any], Dict[str, Any], Dict[str, List[str]]]:
    """
    Load the processed samples, labels, and metadata.

    Args:
        samples_path (str): Path to the processed samples pickle file.
        labels_path (str): Path to the labels pickle file.
        metadata_path (str): Path to the metadata JSON file.

    Returns:
        Tuple[List[Dict[str, Dict]], List[Any], Dict[str, Any], Dict[str, List[str]]]: 
        A tuple containing the list of processed samples, the list of labels, 
        the metadata dictionary (including keys_info), and the keys info dictionary.
    """
    def load_pickle(file_path: str) -> Any:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Successfully loaded data from {file_path}")
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

    def load_json(file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f"Successfully loaded metadata from {file_path}")
            return data
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            raise
        except json.JSONDecodeError:
            print(f"Error: There was a problem decoding the JSON file {file_path}.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while loading {file_path}: {str(e)}")
            raise

    samples = load_pickle(samples_path)
    labels = load_pickle(labels_path)
    metadata = load_json(metadata_path)

    return samples, labels, metadata, metadata['keys_info']

# Example usage:
if __name__ == "__main__":
    samples, labels, metadata, keys_info = load_processed_dataset()
    print(f"Loaded {len(samples)} samples and {len(labels)} labels")
    print(f"Metadata: {metadata}")
    print(f"First sample dense dict: {samples[0]['dense_dict']}")
    print(f"First sample sparse dict: {samples[0]['sparse_dict']}")
    print(f"First label: {labels[0]}")
    print(f"Keys info: {keys_info}")
