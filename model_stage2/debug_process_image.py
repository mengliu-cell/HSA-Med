#!/usr/bin/env python3
import pdb
import json
import os
from io import BytesIO
from PIL import Image

def debug_process_image():
    """Debug the process_image function step by step"""
    
    print("=== DEBUGGING process_image FUNCTION ===")
    
    # Load a sample to get an image path
    with open('/datasets/ms_cxr/train_fixed.jsonl', 'r') as f:
        sample_line = f.readline().strip()
    
    sample_data = json.loads(sample_line)
    image_path = sample_data.get('path', '')
    
    print(f"Sample image path: {image_path}")
    
    # Define parameters
    min_pixels = 262144  # From config
    max_pixels = 4194304  # From config
    
    print(f"min_pixels: {min_pixels}")
    print(f"max_pixels: {max_pixels}")
    
    # Simulate the actual process_image logic step by step
    print(f"\nStep 1: Input image = '{image_path}'")
    
    # Combine with image directory
    full_image_path = os.path.join('/datasets/ms_cxr', image_path)
    print(f"Step 2: Full path = '{full_image_path}'")
    
    # Check if it's a string path
    if isinstance(full_image_path, str):
        print(f"Step 3: It's a string path, checking if file exists...")
        
        # This is where the error occurs
        print(f"Step 4: Attempting to open image at '{full_image_path}'")
        
        # Check if it's a file first
        if not os.path.isfile(full_image_path):
            print(f"ERROR: '{full_image_path}' is not a file!")
            print(f"  - Is it a file? {os.path.isfile(full_image_path)}")
            print(f"  - Does it exist? {os.path.exists(full_image_path)}")
            print(f"  - What is it? {os.path.isdir(full_image_path) and 'DIRECTORY' or 'OTHER'}")
            
            # Check if the path has problematic components
            if '/.' in full_image_path or full_image_path.endswith('/.'):
                print(f"PROBLEM DETECTED: Path contains '/.' or ends with '/.'")
                
                # Show path components
                print(f"Full path: {full_image_path}")
                print(f"Components: {full_image_path.split('/') if '/' in full_image_path else [full_image_path]}")
                
                # Try to fix the path
                fixed_path = full_image_path.replace('/./', '/')
                if fixed_path.endswith('/.'):
                    fixed_path = fixed_path[:-2]
                print(f"Potentially fixed path: {fixed_path}")
                
                print(f"Fixed path exists? {os.path.isfile(fixed_path)}")
        
        else:
            print(f"SUCCESS: File exists at '{full_image_path}'")
    
    print("\n=== INTERACTIVE DEBUG MODE ===")
    print("Available variables:")
    print(f"  - image_path: {image_path}")
    print(f"  - full_image_path: {full_image_path}")
    print(f"  - min_pixels: {min_pixels}")
    print(f"  - max_pixels: {max_pixels}")
    
    pdb.set_trace()

if __name__ == "__main__":
    debug_process_image()
