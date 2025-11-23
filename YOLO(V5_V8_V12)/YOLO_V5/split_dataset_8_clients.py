import os
import shutil
import random
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

# --- Configuration ---
SOURCE_DATA_DIR = 'coco128'
DEST_DATA_DIR = 'Federated_COCO128'
NUM_CLIENTS = 8
CLIENT_VAL_SPLIT_RATIO = 0.2  # 20% of each client's data will be for their local validation
RANDOM_SEED = 42

# Set random seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- Paths ---
# --- Paths ---
source_images_dir = os.path.join(SOURCE_DATA_DIR, 'images', 'train2017')
source_labels_dir = os.path.join(SOURCE_DATA_DIR, 'labels', 'train2017')
source_yaml_file = os.path.join(SOURCE_DATA_DIR, 'data.yaml')

# --- 1. Read Class Names from Source YAML ---
class_names = []
num_classes = 0
if os.path.exists(source_yaml_file):
    with open(source_yaml_file, 'r') as f:
        source_yaml = yaml.safe_load(f)
        class_names = source_yaml.get('names', [])
        num_classes = source_yaml.get('nc', len(class_names))
    print(f"Loaded {num_classes} class names from {source_yaml_file}")
else:
    print(f"Warning: {source_yaml_file} not found. 'nc' and 'names' will be empty in client YAMLs.")

# --- 2. Get All Source Files ---
try:
    all_images = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(all_images)} total images in {source_images_dir}")
except FileNotFoundError:
    print(f"Error: Source directory not found: {source_images_dir}")
    print("Please make sure the 'coco128' dataset is downloaded and in the correct path.")
    exit()
    
# Shuffle the dataset
random.shuffle(all_images)

# --- 3. Split Files Among Clients ---
# Use numpy.array_split to handle potential unequal splits, although 128/8 is clean
client_file_partitions = np.array_split(all_images, NUM_CLIENTS)
print(f"Split dataset into {len(client_file_partitions)} partitions.")

# Clean slate for destination directory
if os.path.exists(DEST_DATA_DIR):
    shutil.rmtree(DEST_DATA_DIR)
os.makedirs(DEST_DATA_DIR, exist_ok=True)
print(f"Created root federated directory: {DEST_DATA_DIR}")

# --- Helper function to copy files ---
def copy_files(file_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    """Copies image and its corresponding label file."""
    for img_file in file_list:
        # Get corresponding label file (e.g., image.jpg -> image.txt)
        base_filename = os.path.splitext(img_file)[0]
        lbl_file = f"{base_filename}.txt"
        
        # Source paths
        src_img_path = os.path.join(src_img_dir, img_file)
        src_lbl_path = os.path.join(src_lbl_dir, lbl_file)
        
        # Destination paths
        dst_img_path = os.path.join(dst_img_dir, img_file)
        dst_lbl_path = os.path.join(dst_lbl_dir, lbl_file)
        
        # Copy files if they exist
        if os.path.exists(src_img_path) and os.path.exists(src_lbl_path):
            shutil.copy(src_img_path, dst_img_path)
            shutil.copy(src_lbl_path, dst_lbl_path)
        else:
            print(f"Warning: Missing file pair for {img_file}. Skipping.")

# --- 4. Process Each Client ---
for i, client_files in enumerate(client_file_partitions):
    client_id = i + 1
    client_name = f'client_{client_id}'
    client_root_dir = os.path.join(DEST_DATA_DIR, client_name)
    
    print(f"\nProcessing {client_name} with {len(client_files)} images...")
    
    # Create client directory structure
    client_train_img_dir = os.path.join(client_root_dir, 'images', 'train')
    client_val_img_dir = os.path.join(client_root_dir, 'images', 'val')
    client_train_lbl_dir = os.path.join(client_root_dir, 'labels', 'train')
    client_val_lbl_dir = os.path.join(client_root_dir, 'labels', 'val')
    
    os.makedirs(client_train_img_dir, exist_ok=True)
    os.makedirs(client_val_img_dir, exist_ok=True)
    os.makedirs(client_train_lbl_dir, exist_ok=True)
    os.makedirs(client_val_lbl_dir, exist_ok=True)

    # Split this client's data into train and val
    client_train_files, client_val_files = train_test_split(
        client_files,
        test_size=CLIENT_VAL_SPLIT_RATIO,
        random_state=RANDOM_SEED
    )
    
    print(f"  -> Splitting into {len(client_train_files)} train and {len(client_val_files)} val files.")

    # Copy train files
    copy_files(client_train_files, source_images_dir, source_labels_dir, client_train_img_dir, client_train_lbl_dir)
    
    # Copy val files
    copy_files(client_val_files, source_images_dir, source_labels_dir, client_val_img_dir, client_val_lbl_dir)

    # --- 5. Create client data.yaml ---
    # Paths in YAML must be relative to the project root (where yolov5/train.py is called)
    # We use forward slashes for cross-platform compatibility (YOLOv5 expects this)
    train_path_rel = os.path.join(client_root_dir, 'images', 'train').replace('\\', '/')
    val_path_rel = os.path.join(client_root_dir, 'images', 'val').replace('\\', '/')

    client_yaml_data = {
        'train': f'./{train_path_rel}',
        'val': f'./{val_path_rel}',
        'nc': num_classes,
        'names': class_names
    }
    
    client_yaml_path = os.path.join(client_root_dir, 'data.yaml')
    with open(client_yaml_path, 'w') as f:
        yaml.dump(client_yaml_data, f, sort_keys=False, default_flow_style=False)
    
    print(f"  -> Created {client_yaml_path}")

print("\n--- Federated Dataset Creation Complete! ---")
print(f"Data for 8 clients is ready in '{DEST_DATA_DIR}' directory.")