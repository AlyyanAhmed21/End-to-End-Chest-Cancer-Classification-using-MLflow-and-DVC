# check_data_balance.py

import os
from pathlib import Path

def check_dataset_balance(data_path: Path):
    """
    Checks and prints the balance of classes in a dataset directory.

    The expected directory structure is:
    - data_path/
        - class_A/
            - image1.jpg
            - image2.jpg
            ...
        - class_B/
            - image1.jpg
            - image2.jpg
            ...
    
    Args:
        data_path (Path): The path to the main dataset directory.
    """
    print(f"--- Checking Dataset Balance at: {data_path} ---\n")

    if not data_path.is_dir():
        print(f"❌ ERROR: The provided path is not a valid directory.")
        return

    class_names = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    if not class_names:
        print("❌ ERROR: No class subdirectories found in the dataset folder.")
        return

    print(f"Found {len(class_names)} classes: {', '.join(class_names)}\n")
    
    class_counts = {}
    total_images = 0

    for class_name in class_names:
        class_dir = data_path / class_name
        # Count files, ignoring subdirectories (like .ipynb_checkpoints)
        num_images = len([f for f in class_dir.iterdir() if f.is_file()])
        class_counts[class_name] = num_images
        total_images += num_images

    print("--- Image Counts per Class ---")
    for class_name, count in class_counts.items():
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        print(f"- {class_name:<20}: {count:>5} images ({percentage:.2f}%)")
    
    print("-" * 35)
    print(f"- {'Total':<20}: {total_images:>5} images\n")
    
    print("--- For your training script ---")
    print("Use these counts to calculate your class_weight dictionary.")


if __name__ == "__main__":
    # --- IMPORTANT ---
    # Update this path to point to your actual dataset folder.
    # This is the folder that contains the 'Normal' and 'adenocarcinoma' subfolders.
    dataset_directory = Path("artifacts/data_ingestion/Chest-CT-Scan-data")
    
    check_dataset_balance(dataset_directory)