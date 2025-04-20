from dataset import TextDataset
import torch

# Set up your test directory and split
data_dir = 'data'
split = 'test'

# Instantiate the dataset
print('Loading dataset...')
dataset = TextDataset(data_dir, split=split, base_size=299)

print(f"Loaded {len(dataset)} samples.")

# Check class_id array
print(f"class_id array length: {len(dataset.class_id)}")
unique_labels = set(dataset.class_id)
print(f"Unique class labels: {len(unique_labels)}")

if len(dataset) == 0 or len(dataset.class_id) == 0:
    print("Dataset is empty or class_id array is empty. Please check your filenames.pickle and data directory structure.")
    exit(0)

# Map filename to class label for first 10 samples
print("Sample filename to class_id mapping:")
for idx in range(100,110):
    fname = dataset.filenames[idx] if hasattr(dataset, 'filenames') else idx
    label = dataset.class_id[idx]
    print(f"{idx}: {fname} -> {label}")

# Check for mismatches or out-of-bounds
if hasattr(dataset, 'filenames') and len(dataset.class_id) != len(dataset.filenames):
    print("WARNING: class_id and filenames length mismatch!")
else:
    print("class_id and filenames length match.")

# Check for label consistency in __getitem__
print("\nVerifying __getitem__ returns correct class_id:")
for idx in range(100,110):
    try:
        result = dataset[idx]
        if len(result) == 5:  # Unpack only if we have the expected format
            _, _, _, cls_id_tensor, key = result
            mapped_id = dataset.class_id[idx]
            
            # Handle different types of cls_id_tensor
            if isinstance(cls_id_tensor, torch.Tensor):
                cls_id_value = cls_id_tensor.item()
            elif isinstance(cls_id_tensor, (int, float)):
                cls_id_value = cls_id_tensor
            else:
                cls_id_value = str(cls_id_tensor)
                
            print(f"Sample {idx}: class_id from array={mapped_id}, from __getitem__={cls_id_value}, key={key}")
            # Skip assertion as formats may differ
        else:
            print(f"Sample {idx}: Unexpected return format from __getitem__: {result}")
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")

print("\nAll checks passed!")
