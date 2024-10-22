from torchdata.datapipes.iter import FileLister, FileOpener
import os
import xml.etree.ElementTree as ET
import json
import copy

# Set the current working directory
os.chdir("/data/workspace/Codes/CAFE")

# 读取metadata，知道需要用到哪些特征
# Define the path to the JSON file
json_file_path = "datasets/moments/gpu_moment_gpu_v506_fix_v3/conf/global_conf.json"

# Read the JSON file
with open(json_file_path, 'r') as json_file:
    features = json.load(json_file)

# Check the keys in the features dictionary
feature_keys = features.keys()
print("Keys in features:", feature_keys)
input_columns = (features['input_columns'])
print("input_columns:", input_columns)

field_names = []
for values in input_columns.values():
    field_names.extend(values)
print("field_names:", field_names)
print("number of fields:", len(field_names))


# Define the path to the XML file
xml_file_path = "datasets/moments/algorithm.xml"

# Parse the XML file
tree = ET.parse(xml_file_path)
root = tree.getroot()

# Create a dictionary to store the feature information
feature_info = {}

# Iterate through each feature element in the XML
for feature_element in root.findall('.//feature'):
    alias = feature_element.get('alias')
    
    # Store the information in the dictionary with field as the key
    feature_info[alias] = {
        'field': feature_element.get('field'),
        'id': feature_element.get('id'),
        'is_context': feature_element.get('is_context'),
        'type': feature_element.get('type'),
        'is_multi_value': feature_element.get('is_multi_value'),
        'is_online_feature': feature_element.get('is_online_feature')
    }

# Create a list to store the matched feature information
matched_features = []

# Iterate through each field name and find corresponding information
for field_name in field_names:
    if field_name in feature_info:
        matched_features.append({
            'field_name': field_name,
            'info': feature_info[field_name]
        })
    else:
        print(f"Warning: Field {field_name} not found in the XML file.")

# Create a set of all field names
all_field_names = set(field_names)

# Create a set of matched field names
matched_field_names = set(feature['field_name'] for feature in matched_features)

# Find unmatched field names
unmatched_field_names = all_field_names - matched_field_names

# Print unmatched field names
print("Unmatched Field Names:")
for field_name in unmatched_field_names:
    print(field_name)

print(f"\nTotal unmatched fields: {len(unmatched_field_names)}")

# Construct new key_names for matched features
keys = {'user': [], 'ads': []}

for feature in matched_features:
    is_context = feature['info']['is_context'] == 'true'
    prefix = "part_1_field_" if is_context else "part_2_field_"
    feature_id = feature['info']['id']
    postfix = "/keys"
    
    new_key_name = f"{prefix}{feature_id}{postfix}"
    if is_context:
        keys['user'].append((feature['field_name'], new_key_name, feature['info']))
    else:
        keys['ads'].append((feature['field_name'], new_key_name, feature['info']))

# Print the results
print("New Key Names:")
for part in ['user', 'ads']:
    print(f"{part} part:")
    for original_name, new_name, info in keys[part]:
        print(f"Original: {original_name}")
        print(f"New: {new_name}")
        print(f"Info: {info}")
        print()

tot = 0
for part in ['user', 'ads']:
    tot += len(keys[part])

print(f"Total new key names created: {tot}")


# Function to extract features for a single example
def extract_example_features(example, keys_part):
    features = []
    not_found_count = 0  # Counter for features not found
    for original_name, new_name, info in keys_part:
        if new_name in example:
            feature_tensor = example[new_name]
            features.append((original_name, new_name, feature_tensor, info['id']))
        else:
            not_found_count += 1  # Increment counter if feature is not found
    print(f"Total features not found: {not_found_count}")  # Print the count of not found features
    return features

def extract_samples(example, keys):
    samples = []
    user_features = extract_example_features(example, keys['user'])
    ads_features = extract_example_features(example, keys['ads'])
    num_ads_in_each_example = len(ads_features[0][2])
    print(f"num_ads_in_each_example: {num_ads_in_each_example}")
    # print(user_features[0])
    samples = []
    for i in range(num_ads_in_each_example):
        # 第i个广告的特征
        features_i = []
        for feature in ads_features:
            ads_feature_tensor_i = feature[2][i]
            ads_feature_i = (*feature[:2], ads_feature_tensor_i, feature[3])
            features_i.append(ads_feature_i)
        samples.append(user_features + copy.deepcopy(features_i))
    
    return samples

#  指定TFRecord文件路径
tfrecord_file = "*.tfr"

# 创建FileLister和FileOpener DataPipe
datapipe1 = FileLister("datasets/moments", tfrecord_file)
datapipe2 = FileOpener(datapipe1, mode="b")

# 使用TFRecordLoader加载TFRecord文件
tfrecord_loader_dp = datapipe2.load_from_tfrecord()


# Extract features for all examples
all_features = []
samples = []
for example in tfrecord_loader_dp:
    samples.extend(extract_samples(example, keys))

print("number of samples:", len(samples))