import json
import os
import random
import shutil

########################################################################################################

# Get raw data file names
raw_data_path = './data/raw'
raw_data = [file for file in os.listdir(os.path.join(raw_data_path)) if file.lower().endswith('.png')]
image_ids = [os.path.splitext(file)[0] for file in raw_data]

# Load json file
json_file_dir = './data/labeled'
old_json_name = 'handwriting_data_info_clean.json'
with open(os.path.join(json_file_dir, old_json_name), 'r', encoding='utf8') as file:
    original_data = json.load(file)
new_data = {
    'info': original_data['info'],
    'images': [],
    'annotations': [],
    'licenses': original_data['licenses']
}

# Since the json file has whole information, we must extract to sync our data
for image in original_data['images']:
    if image['id'] in image_ids:
        # Find the corresponding annotations for the extracted image
        found_annotations = False
        for annotation in original_data['annotations']:
            if annotation['image_id'] == image['id']:
                new_data['annotations'].append(annotation)
                found_annotations = True
                break

        # Only append the image and annotations if annotations are found
        if found_annotations:
            new_data['images'].append(image)

# Write a new json file and we will use it
new_json_name = 'label.json'
with open(os.path.join(json_file_dir, new_json_name), 'w', encoding='utf8') as file:
    json.dump(new_data, file, indent=4, ensure_ascii=False)

########################################################################################################

# Divide Dataset into Train/Validation/Test
# Train 0.8
# Validation 0.1
# Test 0.1
dataset = os.listdir('./data/raw')
random.shuffle(dataset)

n_train = int(len(dataset) * 0.8)
n_val = int(len(dataset) * 0.1)
n_test = int(len(dataset) * 0.1)

# 1040 130 130
print(n_train, n_val, n_test)

train_set = dataset[:n_train]
val_set = dataset[n_train: n_train+n_val]
test_set = dataset[-n_test:]

# Store data ids
train_img2id = dict()
val_img2id = dict()
test_img2id = dict()

for image in new_data['images']:
    if image['file_name'] in train_set:
        train_img2id[image['file_name']] = image['id']
    elif image['file_name'] in val_set:
        val_img2id[image['file_name']] = image['id']
    elif image['file_name'] in test_set:
        test_img2id[image['file_name']] = image['id']

# Store the corresponding annotation: filename -> annotation
train_annotations = dict()
val_annotations = dict()
test_annotations = dict()

train_id2img = {train_img2id[img] : img for img in train_img2id.keys()}
val_id2img = {val_img2id[img] : img for img in val_img2id.keys()}
test_id2img = {test_img2id[img] : img for img in test_img2id.keys()}

for annotation in new_data['annotations']:
    dataset_dir = './data'

    if annotation['image_id'] in train_id2img:
        img = train_id2img[annotation['image_id']]
        train_annotations[img] = annotation
        shutil.copy(os.path.join(dataset_dir, 'raw', img), os.path.join(dataset_dir, 'train'))
    elif annotation['image_id'] in val_id2img:
        img = val_id2img[annotation['image_id']]
        val_annotations[img] = annotation
        shutil.copy(os.path.join(dataset_dir, 'raw', img), os.path.join(dataset_dir, 'val'))
    elif annotation['image_id'] in test_id2img:
        img = test_id2img[annotation['image_id']]
        test_annotations[img] = annotation
        shutil.copy(os.path.join(dataset_dir, 'raw', img), os.path.join(dataset_dir, 'test'))

########################################################################################################

# For lmdb dataset
dataset_dir = './data/'
with open(dataset_dir + 'gt_train.txt', 'w') as gt_file:
    for img, annotation in train_annotations.items():
        gt_file.write("{}\t{}\n".format(os.path.join('train', img), annotation['text']))
        
with open(dataset_dir + 'gt_val.txt', 'w') as gt_file:
    for img, annotation in val_annotations.items():
        gt_file.write("{}\t{}\n".format(os.path.join('val', img), annotation['text']))
        
with open(dataset_dir + 'gt_test.txt', 'w') as gt_file:
    for img, annotation in test_annotations.items():
        gt_file.write("{}\t{}\n".format(os.path.join('test', img), annotation['text']))

########################################################################################################
