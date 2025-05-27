import torch
from PIL import Image
import requests
import json
from tqdm import tqdm
import os
import gc
from argparse import ArgumentParser
import numpy as np
from collections import defaultdict
import pandas as pd
from huggingface_hub import login
from torch.utils.data import Dataset
import yaml
from transformers.image_utils import load_image
from torchvision import transforms
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

# Replace "your_token_here" with your Hugging Face token
login("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")

os.environ["HF_HOME"] = "/ceph/hpc/data/d2024d05-018-users/wenyan/cache"

def read_json(ann_path):
    with open(ann_path, 'r') as f:
        ann = json.load(f)
    return ann

class CaptionDataset(Dataset):
    def __init__(self, ann_path, image_dir):
        self.data = utils.read_json(ann_path)
        self.image_dir = image_dir
        
        self.test_data = self._get_test_images()
        print("Number of unique images: ", len(self.test_data))
        
        
    def __len__(self):
        return len(self.test_data)    
    
    def __getitem__(self, idx):
        image_id = self.test_data[idx]["imgid"]
        image_file = self.test_data[idx]["filename"]
        image_path = os.path.join(self.image_dir, image_file)
        gts = [x["raw"] for x in self.test_data[idx]["sentences"]]
        return {"image_id": image_id, "image_file": image_file, "image_path": image_path, "captions": gts}
    
    def _get_test_images(self):
        test_images = []
        for i in range(len(self.data["images"])):
            if self.data["images"][i]["split"] == "test":
                test_images.append(self.data["images"][i])
        return test_images

def collate_fn(batch):
    # Custom collate function to handle batches
    return {
        'image_path': [item['image_path'] for item in batch],
        'image_id': [item['image_id'] for item in batch],
        'captions': [item['captions'] for item in batch]
    }

def get_eval_loader(annotations_path, image_dir, batch_size=4, num_workers=1, transform=None):
    """
    Helper function to create validation data loader
    """
    dataset = CaptionDataset(
        ann_path=annotations_path,
        image_dir=image_dir
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return loader


def load_qwenvl():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "Qwen/Qwen-VL"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=os.environ["HF_HOME"])
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16,
                                                cache_dir=os.environ["HF_HOME"])
    
    return tokenizer, model

def get_caption_prompt(model_name):
    return "Describe this image using one simple sentences."
    # Unified prompt for all models
    # if model_name == "idefics2":
    #     return "Describe this image."
    # elif model_name == "llava":
    #     return "Describe this image using one or more simple sentences."
    # else:
    #     return "Describe this image."

def process_vision_info(prompts):
    # Placeholder for Qwen-VL specific processing
    # This function would need to be implemented based on Qwen-VL requirements
    image_inputs = []
    video_inputs = None
    
    for prompt in prompts:
        for item in prompt[0]["content"]:
            if item["type"] == "image":
                image_inputs.append(item["image"])
    
    return image_inputs, video_inputs

def get_llava_caption_prompt():
    template = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe what is going on in image? <image> \nASSISTANT:"},
            ],
        },
    ]
    return template

def get_qwen_caption_prompt(image_path):
    resize_transform = transforms.Resize((336, 336))
    template = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": resize_transform(load_image(image_path))},
                {"type": "text", "text": "Describe this image using one simple sentence. Avoid explicit references to the image itself (e.g., 'This image shows...', 'Pictured here is...', 'In this photograph...')."},
                # {"type": "text", "text": "Describe this image using one simple sentence."}
                
            ]
        }
    ]
    return template

def evaluate_captioning(batch, processor, model, args):
    if args.model == "llava":
        images = [load_image(img_path) for img_path in batch['image_path']]
        
        template = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": get_caption_prompt(args.model)},
                ],
            },
        ]
        prompts = [processor.apply_chat_template(template, add_generation_prompt=True) for _ in batch['image_path']]
       
    elif args.model == "idefics2":
        images = [[load_image(img_path)] for img_path in batch['image_path']]
        template = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": get_caption_prompt(args.model)},
                ],
            },
        ]
        prompts = [processor.apply_chat_template(template, add_generation_prompt=True) for _ in batch['image_path']]
       
    elif args.model == "qwen":
        prompts = [get_qwen_caption_prompt(img_path) for img_path in batch['image_path']]
        image_inputs, video_inputs = process_vision_info(prompts)
        prompts = [processor.apply_chat_template(p, add_generation_prompt=True, tokenize=False) for p in prompts]
    
    else:
        print("Model not supported")
        return []
    
    if args.model != "qwen":
        inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt").to(model.device, torch.float16)
    else:
        inputs = processor(images=image_inputs, videos=video_inputs, text=prompts, padding=True, return_tensors="pt").to(model.device, torch.float16)
    
    with torch.no_grad():
        if args.model == "qwen":
            generate_ids = model.generate(**inputs, max_new_tokens=50)  # Set max_new_tokens to 40 as requested
        else:
            generate_ids = model.generate(**inputs, max_new_tokens=128)  # Set max_new_tokens to 40 as requested
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        captions = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return captions

if __name__ == "__main__": 
    argparser = ArgumentParser()
    argparser.add_argument("--out_dir", type=str, default="./flickr30k_results")
    argparser.add_argument("--model", type=str, default="llava", choices=["llava", "idefics2", "qwen"])
    argparser.add_argument("--data_config", type=str, default="./datasets.yaml")
    argparser.add_argument("--batch_size", type=int, default=4)
    argparser.add_argument("--num_workers", type=int, default=1)
    argparser.add_argument("--dataset", type=str, default="flickr30k")
    
    args = argparser.parse_args()
    
    with open(args.data_config, "r") as file:
        config = yaml.safe_load(file)
    
    # Get paths from config file
    
    annotations_path = config["datasets"][args.dataset]["ann_path"]
    image_dir = config["datasets"][args.dataset]["image_dir"]
    
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # Load model
    if args.model == "llava":
        processor, model = utils.load_llava()
        print("Loaded LLaVA model")
    elif args.model == "idefics2":  
        processor, model = utils.load_idefics2()
        print("Loaded IDEFICS2 model")
    elif args.model == "qwen":
        processor, model = utils.load_qwenvl()
        print("Loaded Qwen-VL model")
        # For Qwen-VL specific processing
        from qwen_vl_utils import process_vision_info
    
    # Create evaluation data loader
    eval_loader = get_eval_loader(
        annotations_path=annotations_path, 
        image_dir=image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Evaluation loop
    results = []
    with open(os.path.join(out_dir, f"{args.model}_{args.dataset}_captions.jsonl"), "w") as output_file:
        for batch in tqdm(eval_loader):
            captions = evaluate_captioning(batch, processor, model, args)
            
            for i, caption in enumerate(captions):
                result = {
                    'image_id': batch['image_id'][i],
                    'image_path': batch['image_path'][i],
                    'generated_caption': caption,
                    'reference_captions': batch['captions'][i] if len(batch['captions']) > i else []
                }
                results.append(result)
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Evaluation completed. Results saved to {os.path.join(out_dir, f'{args.model}_{args.dataset}_captions.jsonl')}")
    
    # Calculate metrics if reference captions are available
    # This would typically include BLEU, METEOR, ROUGE-L, CIDEr, etc.
    # You'd need to import the appropriate libraries for these metrics
    
    print("Evaluation complete!")