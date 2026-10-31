import json
import os

def fix_dataset_paths(input_file, output_file, old_prefix, new_prefix):
    """将数据集中的绝对路径转换为相对于挂载点的新路径"""
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            
            # 修改图像路径
            if 'path' in data:
                old_path = data['path']
                if old_path.startswith(old_prefix):
                    # 将原始绝对路径转换为相对于挂载点的新路径
                    relative_path = old_path[len(old_prefix):]
                    if relative_path.startswith('/'):
                        relative_path = relative_path[1:]  # 移除开头的斜杠
                    new_path = os.path.join(new_prefix, relative_path)
                    data['path'] = new_path
            
            outfile.write(json.dumps(data) + '\n')

# 修正训练集
fix_dataset_paths(
    '/datasets/ms_cxr/train.jsonl', 
    '/datasets/ms_cxr/train_fixed.jsonl',
    '/home/ubuntu/medgroundr1_dataset',  # 原始前缀
    '.'  # 新前缀 - 因为图像将与数据文件在同一挂载点下
)

# 修正测试集
fix_dataset_paths(
    '/datasets/ms_cxr/test.jsonl', 
    '/datasets/ms_cxr/test_fixed.jsonl',
    '/home/ubuntu/medgroundr1_dataset',  # 原始前缀
    '.'  # 新前缀
)

print("Dataset paths fixed successfully!")
