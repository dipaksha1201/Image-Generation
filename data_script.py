import os
import pickle

base_dir = 'data/CUB_200_2011'
images_file = os.path.join(base_dir, 'images.txt')
split_file = os.path.join(base_dir, 'train_test_split.txt')

# 1. Load ID to filename mapping
id_to_filename = {}
with open(images_file, 'r') as f:
    for line in f.readlines():
        idx, path = line.strip().split()
        id_to_filename[int(idx)] = path[:-4]  # remove '.jpg'

# 2. Load split info
train_filenames = []
test_filenames = []
with open(split_file, 'r') as f:
    for line in f.readlines():
        idx, is_train = line.strip().split()
        idx = int(idx)
        is_train = int(is_train)
        if is_train == 1:
            train_filenames.append(id_to_filename[idx])
        else:
            test_filenames.append(id_to_filename[idx])

# 3. Save pickles
with open('data/train/filenames.pickle', 'wb') as f:
    pickle.dump(train_filenames, f)
with open('data/test/filenames.pickle', 'wb') as f:
    pickle.dump(test_filenames, f)

print(f"Train samples: {len(train_filenames)}")
print(f"Test samples: {len(test_filenames)}")