from torchdata.datapipes.iter import FileLister, FileOpener
from typing import List, Dict, Tuple
import os
import xml.etree.ElementTree as ET
import json
import copy
import torch
import numpy as np
import pickle
from Sample import Sample


"""============================================
1. 读取metadata，知道需要用到哪些特征
============================================"""

class FeatureConf(object):
    def __init__(self, json_file_path):
        with open(json_file_path, 'r') as json_file:
            self.global_conf = json.load(json_file)
        self.process_columns()
    
    def process_columns(self):
        input_columns = self.global_conf['input_columns']
        self.all_fields = []
        for values in input_columns.values():
            self.all_fields.extend(values)

        self.label_name = 'ltr_ecpm_label'
        self.all_fields.append(self.label_name)

        self.additional_fields = [
            'adgroup_id'
        ]
        self.all_fields.extend(self.additional_fields)

        # self.user_dense_features = input_columns['user_dense']
        self.ad_dense_features = input_columns['ad_dense']
        self.ad_dense_embedding_features = input_columns['ad_dense_embedding']
        self.user_sparse_features = input_columns['user_sparse']
        self.user_sparse_multi_features = input_columns['user_sparse_multi']
        self.ad_sparse_features = input_columns['ad_sparse']
        self.ad_sparse_multi_features = input_columns['ad_sparse_multi']
        # self.user_sparse_query_features = input_columns['user_sparse_query_attention_query']
        # self.user_sparse_sequence_features = input_columns['user_sparse_sequence_block_ad_click_seq']
        # self.sequence_ad_click_seq_length = input_columns['user_dense_sequence_length_block_ad_click_seq']
        self.aux_fields = self.global_conf['aux_fields']

feature_conf = FeatureConf("datasets/moments/gpu_moment_gpu_v506_fix_v3/conf/global_conf.json")

class Feature(object):
    name: str
    dimension: int
    id: int
    is_context: bool
    is_ext_info: bool
    new_key_name: str
    default: Dict[str, str]

    def __init__(self, name, dimension, id, is_context, is_ext_info):
        self.name = name
        self.dimension = dimension
        self.id = id
        self.is_context = is_context
        self.is_ext_info = is_ext_info
        self.new_key_name = self.get_new_key_name()

    def get_new_key_name(self):
        if self.is_ext_info:
            return f"ad_ext/{self.name}"
        elif self.is_context:
            return f"part_1_field_{self.id}/keys"
        else:
            return f"part_2_field_{self.id}/keys"

class AlgFeatInfo(object):
    def __init__(self):
        pass

    def load_from_xml(self, xml_file_path):
        self.xml_file_path = xml_file_path
        self.features = {}
        self.process_features()

    def process_features(self):
        tree = ET.parse(self.xml_file_path)
        root = tree.getroot()
        for feature_element in root.findall('.//feature'):
            alias = feature_element.get('alias')
            feature = Feature(
                name=alias,
                dimension=feature_element.get('feature_dimension'),
                id=feature_element.get('id'),
                is_context=feature_element.get('is_context') == 'true',
                is_ext_info=feature_element.get('is_ext_info') == 'true'
            )
            
            
            # Check for default value
            default_element = feature_element.find('default')
            if default_element is not None:
                default_type = default_element.get('type')
                default_values = default_element.get('values')
                feature.default = {
                    'type': default_type,
                    'values': default_values
                }
            
            self.features[alias] = feature

    def get_feature(self, alias: str):
        return self.features[alias]
    
    def get_feature_default(self, alias: str):
        return self.features[alias].default

    def get_matched_features(self, field_names):
        matched_features = []
        for field_name in field_names:
            if field_name in self.features:
                matched_features.append(self.features[field_name])
            else:
                print(f"Warning: Field {field_name} not found in the XML file.")
        return matched_features

    # def get_unmatched_fields(self, field_names):
    #     all_field_names = set(field_names)
    #     matched_field_names = set(self.features.keys())
    #     return all_field_names - matched_field_names

# Initialize FeatureInfo
alg_feat_info = AlgFeatInfo()
alg_feat_info.load_from_xml("datasets/moments/algorithm.xml")

# Get matched features
matched_features = alg_feat_info.get_matched_features(feature_conf.all_fields)

# # Get unmatched fields
# unmatched_fields = feature_info.get_unmatched_fields(feature_conf.all_fields)

# print("Unmatched Field Names:")
# for field_name in unmatched_fields:
#     print(field_name)

# print(f"\nTotal unmatched fields: {len(unmatched_fields)}")

"""============================================
2. 构造新的key_names
============================================"""
keys = {
    'user': [], 
    'ads': [], 
    'ext': [], 
    'label': [], 
    'additional': []
}

for feature in matched_features:
    new_key_name = feature.get_new_key_name()
    if feature.is_ext_info:
        if feature.name == feature_conf.label_name:
            keys['label'].append((feature.name, new_key_name, feature))
        elif feature.name in feature_conf.additional_fields:
            keys['additional'].append((feature.name, new_key_name, feature))
        else:
            keys['ext'].append((feature.name, new_key_name, feature))
    elif feature.is_context:
        keys['user'].append((feature.name, new_key_name, feature))
    else:
        keys['ads'].append((feature.name, new_key_name, feature))

# Print the results
# print("New Key Names:")
# for part in keys.keys():
#     print(f"{part} part:")
#     for original_name, new_name, feature in keys[part]:
#         print(f"Original: {original_name}")
#         print(f"New: {new_name}")
#         print(f"Info: {feature.__dict__}")
#         print()

tot = sum(len(keys[part]) for part in ['user', 'ads', 'ext'])
print(f"Total new key names created: {tot}")

# Function to extract features for a single example
def extract_example_features(example, keys_part):
    features = {}
    not_found_count = 0  # Counter for features not found
    for original_name, new_name, feature in keys_part:
        if new_name in example:
            feature_tensor = example[new_name]
            features[original_name] = (new_name, feature_tensor, feature.id)
        else:
            not_found_count += 1  # Increment counter if feature is not found
    print(f"Total features not found: {not_found_count}")  # Print the count of not found features
    return features

def process_labels(labels):
    assert len(labels) == 1, "There should be only one label"
    labels: list[torch.Tensor] = list(labels.values())[0][1]
    labels = torch.tensor(labels, dtype=torch.int)
    # Set a threshold
    top = 10
    # Convert labels to binary based on the threshold
    binary_labels = torch.where(labels <= top, torch.ones_like(labels), torch.zeros_like(labels))
    return binary_labels

def extract_samples(example, keys):
    samples = []
    user_features = extract_example_features(example, keys['user'])
    ads_features = extract_example_features(example, keys['ads'])
    ext_features = extract_example_features(example, keys['ext'])
    additional = extract_example_features(example, keys['additional'])
    labels = extract_example_features(example, keys['label'])

    binary_labels = process_labels(labels)

    # Use the first ads feature to determine the number of ads
    first_ads_feature = next(iter(ads_features.values()))
    num_ads_in_each_example = len(first_ads_feature[1])
    
    # Assert that all ads_features and ext_features have the same number of elements
    assert all(len(feature[1]) == num_ads_in_each_example for feature in ads_features.values()), "Not all ads_features have the same number of elements"
    assert all(len(feature[1]) == num_ads_in_each_example for feature in ext_features.values()), "Not all ext_features have the same number of elements"
    print(f"num_ads_in_each_example: {num_ads_in_each_example}")

    # assert all adgroup_id are not 0
    assert all(value != 0 for value in additional['adgroup_id'][1]), "All adgroup_id is non-zero"

    samples = []
    for i in range(num_ads_in_each_example):
        # 第i个广告的特征
        features_i = user_features.copy()
        for original_name, (new_name, feature_tensor, feature_id) in ads_features.items():
            features_i[original_name] = (new_name, feature_tensor[i], feature_id)
        for original_name, (new_name, feature_tensor, feature_id) in ext_features.items():
            features_i[original_name] = (new_name, feature_tensor[i], feature_id)
        samples.append(features_i)
    
    return samples, binary_labels

#  指定TFRecord文件路径
tfrecord_file = "*.tfr"

# 创建FileLister和FileOpener DataPipe
datapipe1 = FileLister("datasets/moments", tfrecord_file)
datapipe2 = FileOpener(datapipe1, mode="b")

# 使用TFRecordLoader加载TFRecord文件
tfrecord_loader_dp = datapipe2.load_from_tfrecord()

# Extract features for all examples
samples = []
labels = torch.Tensor([])
for example in tfrecord_loader_dp:
    samples_i, labels_i = extract_samples(example, keys)
    samples.extend(samples_i)
    labels = torch.cat((labels, labels_i), dim=0)

print("number of samples:", len(samples))

print("first sample:", samples[0])
print("label of first sample:", labels[0])

class SampleParser:
    def __init__(self, feature_conf: FeatureConf, alg_feat_info: AlgFeatInfo):
        self.feature_conf = feature_conf
        self.alg_feat_info = alg_feat_info
        self._fill_keys()

    def _fill_keys(self):
        self.dense_keys = self.feature_conf.ad_dense_features + self.feature_conf.ad_dense_embedding_features
        self.single_sparse_keys = self.feature_conf.user_sparse_features + self.feature_conf.ad_sparse_features
        self.multi_sparse_keys = self.feature_conf.user_sparse_multi_features + self.feature_conf.ad_sparse_multi_features

    def parse_sample(self, sample: Dict[str, Tuple[str, torch.Tensor, int]]) -> Dict[str, Dict]:
        dense_dict = self.parse_dense(sample)
        sparse_dict = self.parse_sparse(sample)
        return {"dense_dict": dense_dict, "sparse_dict": sparse_dict}

    def parse_dense(self, sample: Dict[str, Tuple[str, torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
        dense_dict = {}
        for key in self.dense_keys:
            if key in sample:
                dense_dict[key] = sample[key][1]
            else:   # 如果sample中没有这个key，则使用default value
                feature = self.alg_feat_info.get_feature(key)
                if feature and hasattr(feature, 'default'):
                    default = feature.default
                    default_type = default['type']
                    default_values = default['values']
                    if default_type == 'FLOAT':
                        default_value = torch.tensor([float(v) for v in default_values.split(',')])
                    elif default_type == 'INT':
                        default_value = torch.tensor([int(v) for v in default_values.split(',')])
                    else:
                        raise ValueError(f"Unsupported default type: {default_type}")
                    
                    dense_dict[key] = default_value
                else:
                    raise ValueError(f"No default value found for key: {key}")
        return dense_dict

    def parse_sparse(self, sample: Dict[str, Tuple[str, torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
        sparse_dict = {}
        for key, (new_name, value, feature_id) in sample.items():
            if key in self.single_sparse_keys or key in self.multi_sparse_keys:
                sparse_dict[key] = value
        return sparse_dict

    def get_keys_info(self):
        return {
            "dense_keys": self.dense_keys,
            "single_sparse_keys": self.single_sparse_keys,
            "multi_sparse_keys": self.multi_sparse_keys
        }

sample_parser = SampleParser(feature_conf, alg_feat_info)
samples_parsed = [sample_parser.parse_sample(sample) for sample in samples]

print(samples_parsed[0]['dense_dict'])
print(samples_parsed[0]['sparse_dict'])

# Create a directory to store the dataset if it doesn't exist
os.makedirs('datasets/moments/processed', exist_ok=True)

# Save the parsed samples
with open('datasets/moments/processed/samples_parsed.pkl', 'wb') as f:
    pickle.dump(samples_parsed, f)

# Save the labels
with open('datasets/moments/processed/labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

print("Dataset saved successfully.")

# Save the keys information
keys_info = sample_parser.get_keys_info()

# Save the keys information
with open('datasets/moments/processed/keys_info.json', 'w') as f:
    json.dump(keys_info, f, indent=2)

print("Keys information saved successfully.")
