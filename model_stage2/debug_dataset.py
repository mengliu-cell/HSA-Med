import json
import os

def debug_dataset_issues(dataset_file):
    """Debug issues in the dataset file"""
    print(f"Debugging dataset file: {dataset_file}")
    
    if not os.path.exists(dataset_file):
        print(f"ERROR: Dataset file does not exist: {dataset_file}")
        return
    
    problematic_entries = []
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                if 'path' in data:
                    path = data['path']
                    
                    # Check for common issues
                    if path == '/':
                        problematic_entries.append((line_num, path, "Root directory path"))
                    elif path.startswith('//'):
                        problematic_entries.append((line_num, path, "Double slash path"))
                    elif path == '.':
                        problematic_entries.append((line_num, path, "Current directory path"))
                    elif path.startswith('./'):
                        problematic_entries.append((line_num, path, "Relative path starting with ./"))
                    elif path.startswith('../'):
                        problematic_entries.append((line_num, path, "Parent directory path"))
                    elif path == '':
                        problematic_entries.append((line_num, path, "Empty path"))
                    elif path == '/datasets/ms_cxr/.':  # This is likely the problem
                        problematic_entries.append((line_num, path, "Combined path issue: /datasets/ms_cxr/."))
                    else:
                        # Check if path exists when combined with base dir
                        # Clean the path to remove any leading slash if we're going to join it
                        clean_path = path.lstrip('/')
                        full_path = os.path.join('/datasets/ms_cxr', clean_path)
                        
                        # Special check for the problematic case
                        if '/.' in path or path.endswith('/.'):
                            problematic_entries.append((line_num, path, f"Path contains '/.' or ends with '/.': {path}"))
                        elif not os.path.isfile(full_path):
                            if not path.startswith('/') and len(path) > 1:  # Not an absolute path and not empty
                                if path != '.':
                                    problematic_entries.append((line_num, path, f"File does not exist: {full_path}"))
                
                if line_num <= 10:  # Print first 10 entries for inspection
                    print(f"Line {line_num}: path = '{data.get('path', 'NO PATH KEY')}'")
                    
                if line_num % 1000 == 0:
                    print(f"Processed {line_num} lines...")
                    
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
                problematic_entries.append((line_num, line_num, f"Parsing error: {e}"))
    
    print(f"\nFound {len(problematic_entries)} problematic entries:")
    for line_num, path, issue in problematic_entries[:20]:  # Show first 20 issues
        print(f"  Line {line_num}: '{path}' - {issue}")
    
    if len(problematic_entries) > 20:
        print(f"  ... and {len(problematic_entries) - 20} more issues")

# Debug both original and fixed datasets
print("Checking original train dataset:")
debug_dataset_issues('/datasets/ms_cxr/train.jsonl')
print("\n" + "="*50 + "\n")

print("Checking original test dataset:")
debug_dataset_issues('/datasets/ms_cxr/test.jsonl')
print("\n" + "="*50 + "\n")

# Only check fixed datasets if they exist
if os.path.exists('/datasets/ms_cxr/train_fixed.jsonl'):
    print("Checking fixed train dataset:")
    debug_dataset_issues('/datasets/ms_cxr/train_fixed.jsonl')
    print("\n" + "="*50 + "\n")

if os.path.exists('/datasets/ms_cxr/test_fixed.jsonl'):
    print("Checking fixed test dataset:")
    debug_dataset_issues('/datasets/ms_cxr/test_fixed.jsonl')
