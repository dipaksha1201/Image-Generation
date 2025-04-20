import pickle
import os
import sys

DATA_DIR = 'data/test'
PICKLE_FILE = os.path.join(DATA_DIR, 'class_info.pickle')

def main():
    if not os.path.isfile(PICKLE_FILE):
        print(f"File not found: {PICKLE_FILE}")
        sys.exit(1)
    
    with open(PICKLE_FILE, 'rb') as f:
        try:
            class_info = pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle: {e}")
            sys.exit(1)
    
    print(f"Loaded class_info.pickle with type: {type(class_info)}")
    if isinstance(class_info, dict):
        print(f"Keys: {list(class_info.keys())[:10]}")
        for k in list(class_info.keys())[:3]:
            print(f"Key: {k} -> Value: {class_info[k]}")
    elif isinstance(class_info, (list, tuple)):
        print(f"Length: {len(class_info)}")
        print(f"First 10 elements: {class_info[:10]}")
        try:
            import collections, numpy as np
            counter = collections.Counter(class_info)
            print(f"Unique values: {len(counter)}")
            print(f"Value counts: {counter}")
            arr = np.array(class_info)
            print(f"Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.3f}")
        except ImportError:
            print("Install numpy for more statistics.")
    else:
        print(f"Contents: {class_info}")

if __name__ == '__main__':
    main()
