# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from transformers import AutoTokenizer, AutoModel
# import torch
# # default: Load the model on the available device(s)
# # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
# #     "/home/ubuntu/pretrained_model/qwen25_vl_7b_med", torch_dtype="auto", device_map="auto"
# # )
# path = "/home/ubuntu/pretrained_model/intervl25_2b"
# model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     use_flash_attn=True,
#     trust_remote_code=True).eval().cuda()

# # default processer
# # processor = AutoProcessor.from_pretrained("/home/ubuntu/pretrained_model/qwen25_vl_7b_med")
# processor = AutoTokenizer.from_pretrained(path)

# # The default range for the number of visual tokens per image in the model is 4-16384.
# # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# # min_pixels = 256*28*28
# # max_pixels = 1280*28*28
# # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
# def generate_bbox(file_path, question, sentence):
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": f"{file_path}"},
#                 {"type": "text", "text": f"Give you an image and a sentence of the lesion:{sentence}. The size of image: Width:256, Height:256. Please provide the bounding box coordinate of the sentence. [x1, y1, x2, y2] is expected as the output format."},
#                 # {"type": "text", "text": f"Give you an image, the size of image: Width:256, Height:256. Please provide a tight bounding box coordinate of lesion location. [x1, y1, x2, y2] is expected as the output format."},
#             ],
#         }
#     ]

#     # Preparation for inference
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to("cuda")

#     # Inference: Generation of the output
#     generated_ids = model.generate(**inputs, max_new_tokens=512)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
#     print(output_text)
#     return output_text[0]


# if __name__ == "__main__":
#     import json
#     import pandas as pd 
#     from tqdm import tqdm
#     import os
#     json_path = "/home/ubuntu/medgroundr1_dataset/train_fixed.jsonl"
#     results = []
#     with open(json_path, "r") as f:
#         for line in tqdm(f, desc="Processing JSON lines"):
#             if line.strip():  # Skip empty lines
#                 data = json.loads(line)
#                 # print(data)
#                 file_path = os.path.join("/home/ubuntu/medgroundr1_dataset", data["path"])
#                 question = data["questions"][0]
#                 bbox_gt = data["bbox"][0]
#                 sentence = data["label_text"][0]
#                 bbox = generate_bbox(file_path, question, sentence)
#                 results.append({
#                     "file_path": file_path,
#                     "question": question,
#                     "pred_bbox": bbox,
#                     "gt_bbox": bbox_gt
#                 })
#                 print(f"Predicted BBox: {bbox}, Ground Truth BBox: {bbox_gt}")

#     df = pd.DataFrame(results)
#     df.to_csv("bbox_predictions_intervl_2b.csv", index=False)
#     print(f"Saved {len(df)} predictions to bbox_predictions_intervl_2b.csv")



#     # file_path = "/home/ubuntu/data/ms_cxr/train/images/00000001.png"
#     # question = "What is the location of the lesion?"
#     # generate_bbox(file_path, question)




import numpy as np
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

path = "/home/ubuntu/pretrained_model/intervl25_2b"
# path = "OpenGVLab/InternVL2_5-2B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    local_files_only=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


# single-image single-round conversation (单图单轮对话)

if __name__ == "__main__":
    import json
    import pandas as pd 
    from tqdm import tqdm
    import os
    import pandas as pd

    
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    json_path = "/home/ubuntu/medgroundr1_dataset/chestxray8_output_train_fixed_with_solution.jsonl"
    results = []
    with open(json_path, "r") as f:
        for line in tqdm(f, desc="Processing JSON lines"):
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                # print(data)
                file_path = os.path.join("/home/ubuntu/medgroundr1_dataset", data["image"])
                question = data["questions"][0]     
                bbox_gt = data["bbox"][0]
                bbox = [int(float(x)*(448/640)) for x in bbox_gt]
                sentence = data["label_text"][0]
                pixel_values = load_image(file_path, max_num=12).to(torch.bfloat16).cuda()
                generation_config = dict(max_new_tokens=2048, do_sample=True)
                question = f'<image>\nPlease locate the lesion described in the image:{sentence}. Output format:[x1, y1, x2, y2].'
                # print(model.__class__.__module__)
                # print(type(model.language_model))
                # print(hasattr(model.language_model, "generate"))
                response = model.chat(tokenizer, pixel_values, question, generation_config)

                # print(f'User: {question}\nAssistant: {response}')
                # print(f"Predicted BBox: {bbox}, Ground Truth BBox: {bbox_gt}")
                results.append({
                    # "file_path": file_path,
                    "question": question,
                    "pred_bbox": response,
                    "gt_bbox": bbox
                })
        results = pd.DataFrame(results)
        results.to_csv("bbox_predictions_chestxray_intervl_2b.csv", index=False)

