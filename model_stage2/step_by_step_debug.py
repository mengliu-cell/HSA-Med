#!/usr/bin/env python3
import pdb
import json
import os

def step_by_step_debug():
    """Step by step debugging of image path processing"""
    print("Starting step-by-step debugging...")
    
    # Step 1: Load a sample from the dataset
    print("\nStep 1: Loading sample from dataset...")
    with open('/datasets/ms_cxr/train_fixed.jsonl', 'r') as f:
        sample_line = f.readline().strip()
    
    sample_data = json.loads(sample_line)
    print(f"Loaded sample data keys: {list(sample_data.keys())}")
    print(f"Image path in sample: {sample_data.get('path', 'NOT FOUND')}")
    
    # Step 2: Simulate the dataset loading process
    print("\nStep 2: Simulating dataset loading process...")
    image_key = "path"  # This corresponds to data.image_key=path in the command
    image_dir = "/datasets/ms_cxr"  # This corresponds to data.image_dir=/datasets/ms_cxr
    
    print(f"image_key: {image_key}")
    print(f"image_dir: {image_dir}")
    
    # Step 3: Extract images from example
    print("\nStep 3: Extracting images from example...")
    images = sample_data[image_key]
    print(f"Raw image path from example: {images}")
    
    # Step 4: Apply image_dir if provided
    print("\nStep 4: Applying image_dir transformation...")
    if image_dir is not None and isinstance(images, str):  # image paths
        full_image_path = os.path.join(image_dir, images) if not images.startswith('/') else images
        print(f"Full image path after joining with image_dir: {full_image_path}")
    else:
        full_image_path = images
        print(f"Images not transformed: {full_image_path}")
    
    # Step 5: Check if file exists
    print("\nStep 5: Checking if image file exists...")
    if isinstance(full_image_path, str):
        file_exists = os.path.isfile(full_image_path)
        print(f"File exists: {file_exists}")
        print(f"File path checked: {full_image_path}")
        
        if not file_exists:
            print("ERROR: Image file does not exist!")
            print(f"Looking for: {full_image_path}")
            # Check parent directory
            parent_dir = os.path.dirname(full_image_path)
            if os.path.exists(parent_dir):
                print(f"Parent directory exists: {parent_dir}")
                print("Files in parent directory:")
                for f in os.listdir(parent_dir)[:10]:  # Show first 10 files
                    print(f"  {f}")
            else:
                print(f"Parent directory does not exist: {parent_dir}")
    
    # Step 6: Show actual directory structure around the expected path
    print("\nStep 6: Checking actual directory structure...")
    expected_path_parts = images.split('/')
    current_path = "/datasets/ms_cxr"
    
    print(f"Expected image path components: {expected_path_parts}")
    
    for i, part in enumerate(expected_path_parts[:-1]):  # All parts except the last (filename)
        current_path = os.path.join(current_path, part)
        print(f"Level {i}: Path = {current_path}")
        if os.path.exists(current_path):
            print(f"  ✅ Directory exists")
            if os.path.isdir(current_path):
                sub_items = os.listdir(current_path)[:5]  # Show first 5 items
                print(f"  Contents: {sub_items}")
        else:
            print(f"  ❌ Directory does NOT exist!")
            print(f"  Available in parent: {os.listdir(os.path.dirname(current_path)) if os.path.exists(os.path.dirname(current_path)) else 'N/A'}")
            break
    
    # Step 7: Final check with the actual file
    final_path = os.path.join("/datasets/ms_cxr", images)
    print(f"\nStep 7: Final path check: {final_path}")
    print(f"Final path exists: {os.path.isfile(final_path)}")
    
    # Enter interactive debugger here if needed
    print("\nEntering interactive mode...")
    print("You can now inspect variables:")
    print("  - sample_data: the loaded sample")
    print("  - images: the raw image path")
    print("  - image_dir: the image directory")
    print("  - full_image_path: the combined path")
    print("  - final_path: the final path to check")
    
    pdb.set_trace()  # This will pause execution for interactive debugging

if __name__ == "__main__":
    step_by_step_debug()
