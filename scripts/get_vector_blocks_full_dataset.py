import torch
from PIL import Image
import requests
import json
from tqdm import tqdm
from transformers.image_utils import load_image
import os
import gc
from argparse import ArgumentParser
import numpy as np
from collections import defaultdict
import pandas as pd
from huggingface_hub import login
from datasets import Dataset
from vector_dataset import FlatFileVectorBlockStore
import utils
from torch.utils.data import Dataset
import yaml
import pickle
from torchvision import transforms
from glob import glob


# Replace "your_token_here" with your Hugging Face token
login("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")


def load_chameleon():
    from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
    processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf",
                                                   cache_dir=os.environ["HF_HOME"])
    model = ChameleonForConditionalGeneration.from_pretrained("leloy/Anole-7b-v0.1-hf", 
                                                              cache_dir=os.environ["HF_HOME"],
                                                              torch_dtype=torch.bfloat16, device_map="cuda")
    return processor, model

def load_llava():
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    model_id = "llava-hf/llava-1.5-7b-hf"

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=os.environ["HF_HOME"])
    model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16,
                                                        cache_dir=os.environ["HF_HOME"])

    return processor, model

def get_llava_prompt(question):
    template = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}, # here we use the first question for the image, but it does not impact the embeddings
            ],
        },
    ]
    
    prompt = processor.apply_chat_template(template, add_generation_prompt=True)
    return prompt

def get_qwen_prompt(question, image_path):
    resize_transform = transforms.Resize((336, 336))
    template = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": resize_transform(load_image(image_path)),
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    return template

def get_llava_vqa_prompt(question):
    # following the prompt used here at https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md
    template = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question + "\nAnswer the question using a single word or phrase."},
            ],
        },
    ]
    
    prompt = processor.apply_chat_template(template, add_generation_prompt=True)
    return prompt

def qwen_extract(input_batch, processor, model):
    if 'question' in input_batch:
        templates = [get_qwen_prompt(q, img) for q, img in zip(input_batch["question"][0], input_batch["image_path"])]
    else:
        templates = [get_qwen_prompt("What do you see in this image", img) for img in input_batch["image_path"]]
    texts = [processor.apply_chat_template(template, tokenize=False, add_generation_prompt=True) for template in templates]
    image_inputs, video_inputs = process_vision_info(templates)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    hidden_states_before = []
    hidden_states_after = []
            
    def get_hidden_states_after(module, input, output):
        # print(input[0].size(), output.size()) # torch.Size([504, 1280]) torch.Size([126, 3584])
        hidden_states_before.append(input[0])
        hidden_states_after.append(output)
        
    connector_hook = model.visual.merger.register_forward_hook(get_hidden_states_after)
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=30)
        # generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        # response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    connector_hook.remove()
    # print("hidden_states_before: ", hidden_states_before[0].size())
    # print("hidden_states_after: ", hidden_states_after[0].size())
    
    hidden_dim_before = hidden_states_before[0].size(-1) #1280
    hidden_dim_after = hidden_states_after[0].size(-1) #3584
    
    pre_projection = hidden_states_before[0].view(len(input_batch["image_path"]), -1, hidden_dim_before).float().cpu().detach()
    post_projection = hidden_states_after[0].view(len(input_batch["image_path"]), -1, hidden_dim_after).float().cpu().detach()
    return pre_projection, post_projection 
    
def llava_extract(input_batch, processor, model):
    images = [load_image(img_path) for img_path in input_batch["image_path"]]
    if 'question' in input_batch:
        prompts = [get_llava_vqa_prompt(q) for q in input_batch["question"][0]]
    else:
        prompts = [get_llava_vqa_prompt("What do you see in this image") for i in range(len(input_batch["image_path"]))]
 
    inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt").to(model.device, torch.float16)
    
    hidden_states_before = []
    hidden_states_after = []
            
    def get_hidden_states_after(module, input, output):
        hidden_states_before.append(input[0])
        hidden_states_after.append(output)

    # Register the hooks
    connector_hook = model.multi_modal_projector.register_forward_hook(get_hidden_states_after)
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=30)
    connector_hook.remove()
    
    pre_projection = hidden_states_before[0].float().cpu().detach() #.numpy()
    post_projection = hidden_states_after[0].float().cpu().detach() #.numpy()
    return pre_projection, post_projection # 2, 576, 1024/4096
    
    
def idefics2_extract(input_batch, processor, model):
    # image = load_image(image_path)
    # images = [image]
    # prompt = img2q[image_id][1][0] + "<image>"  # get the first question for the image
    
    # batched inputs
    # images = [load_image(img_path) for img_path in input_batch["image_path"]]
    # prompts = [question + "<image>" for question in input_batch["question"][0]]
    if 'question' in input_batch:
        prompts = [get_llava_vqa_prompt(q) for q in input_batch["question"][0]]
    else:
        prompts = [get_llava_vqa_prompt("What do you see in this image") for i in range(len(input_batch["image_path"]))]
        
    images = [[load_image(img_path)] for img_path in input_batch["image_path"]]

    # fix input images to be the same size
    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Create lists to store the hidden states
    hidden_states_before = []
    hidden_states_after = []


    def get_hidden_states_after(module, input, output):
        hidden_states_before.append(input[0])
        hidden_states_after.append(output)

        # Register the hooks
    # vision_hook = model.model.vision_model.register_forward_hook(get_hidden_states)
    connector_hook = model.model.connector.register_forward_hook(get_hidden_states_after)

    # Forward pass to get outputs
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=30)
        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        # response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
    connector_hook.remove()

    pre_projection = hidden_states_before[0].float().cpu().detach() #.numpy()
    post_projection = hidden_states_after[0].float().cpu().detach() #.numpy()
    return pre_projection, post_projection



def run_chameleon_example(processor, model, text, image):
    # prepare image and text prompt
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    prompt = "What do you see in this image?<image>"


    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

    print("vqgan embeddings: ", model.model.vqmodel.encode(inputs['pixel_values']).shape)
    print("bpe embeddings: ", model.model.get_image_tokens(inputs['pixel_values']).shape)
    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=50)
    print(processor.decode(output[0], skip_special_tokens=True))
    
def llava_save_batch_res(batch):
    pass

class CUBDataset(Dataset):
    def __init__(self, image_dir):
        self.images = sorted(glob(os.path.join(image_dir,"*.*/*.png")))
        self.image_dir = image_dir
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_str, image_id = self._get_label(image_path)
        return {"image_id": image_id, "image_path": image_path, "label_str": label_str, "block_id": idx}
    
    def _get_label(self, image_path):
        name_strs = image_path.split("/")
        label_str, image_id = name_strs[-2], name_strs[-1]
        return label_str, image_id


class SeedBenchDataset(Dataset):
    def __init__(self, ann_path, image_dir):
        self.data = utils.read_json(ann_path)
        self.image_dir = image_dir
        self.img2q = defaultdict(list)
        for item in self.data["questions"]:
            if item["question_type_id"] <= 9:
                self.img2q[item["data_id"]].append((item["question"], item["question_id"])) # data_id is the image_id for seed-bench
        self.img2q_list = list(self.img2q.items())
        print("Number of unique images: ", len(self.img2q_list))
        
    def __len__(self):
        return len(self.img2q_list)
    
    def __getitem__(self, idx):
        image_id = self.img2q_list[idx][0]
        image_file = str(image_id)
        image_path = os.path.join(self.image_dir, image_file)
        question = self.img2q[image_id][0] # get the first question here --- this does not impact the embeddings tho
        block_id = idx
        return {"image_id": image_id, "image_file": image_file, "image_path": image_path, "question": question, "block_id": block_id} 

class FoodieQADataset(Dataset):
    def __init__(self, ann_path, image_dir):
        self.data = utils.read_json(ann_path)
        self.image_dir = image_dir
        self.img2q = defaultdict(list)
        for item in self.data:
            self.img2q[item["food_meta"]["food_file"]].append((item["question_en"], item["question_id"]))
        self.img2q_list = list(self.img2q.items())
        print("Number of unique images: ", len(self.img2q_list))   
    
    def __len__(self):
        return len(self.img2q_list)
    
    def __getitem__(self, idx):
        image_file = self.img2q_list[idx][0]
        image_path = os.path.join(self.image_dir, image_file)
        question = self.img2q[image_file][0]
        block_id = idx
        return {"image_file": image_file, "image_path": image_path, "question": question, "block_id": block_id}
    
class VizWizGVQA(Dataset):
    def __init__(self, ann_path, image_dir):
        self.data = utils.read_json(ann_path)
        self.image_dir = image_dir
        self.img2q = defaultdict(list)
        for image_file, ann in self.data.items():
            question_id = int(image_file.split(".")[0].split("_")[-1])
            self.img2q[image_file].append((ann["question"], question_id))
        self.img2q_list = list(self.img2q.items())
        print("Number of unique images: ", len(self.img2q_list))
        
    def __len__(self):
        return len(self.img2q_list)
    
    def __getitem__(self, idx):
        image_file = self.img2q_list[idx][0]
        image_path = os.path.join(self.image_dir, image_file)
        question = self.img2q[image_file][0]
        block_id = idx
        return {"image_file": image_file, "image_path": image_path, "question": question, "block_id": block_id}
    
class Flickr30k(Dataset):
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
        block_id = image_id
        return {"image_id": image_id, "image_file": image_file, "image_path": image_path, "block_id": block_id}
    
    def _get_test_images(self):
        test_images = []
        for i in range(len(self.data["images"])):
            if self.data["images"][i]["split"] == "test":
                test_images.append(self.data["images"][i])
        return test_images
        

if __name__ == "__main__": 
    argparser = ArgumentParser()
    argparser.add_argument("--data_config", type=str, 
                           default="/ceph/hpc/data/d2024d05-018-users/wenyan/code/VisU/datasets.yaml")
    argparser.add_argument("--out_dir", type=str, 
                           default="/ceph/hpc/data/d2024d05-018-users/wenyan/data/VisU/output/embed/llava_vector_blocks_seedbench")
    argparser.add_argument("--model", choices=["llava", "idefics2", "qwen"], default="llava")
    argparser.add_argument("--avg_pool", type=bool, default=False)
    argparser.add_argument("--batch_size", type=int, default=2)
    argparser.add_argument("--dataset", choices=["seedbench", "foodieqa", "vizwizgvqa", "cub", "flickr30k", "coco_test"], default="seedbench")
    args = argparser.parse_args()
    
    out_dir = args.out_dir
    
    ### load yaml file and dataset annotations
    with open(args.data_config, "r") as file:
        config = yaml.safe_load(file)
    
    dataset_name = args.dataset
    if dataset_name in config["datasets"]:
        dataset_config = config["datasets"][dataset_name]
        ann_path = dataset_config.get("ann_path")
        image_dir = dataset_config["image_dir"]
        print(f"Dataset: {dataset_name}")
        print(f"Annotation Path: {ann_path}")
        print(f"Image Directory: {image_dir}")
    else:
        print(f"Error: Dataset '{dataset_name}' not found in the configuration file.")
    
    model_name = args.model
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # load vqav2 validation set
    if ann_path is not None:
        data = utils.read_json(ann_path)
    # vqav2_val = utils.read_json(ann_path)
    
    if model_name == "chameleon":
        processor, model = load_chameleon()
    elif model_name == "idefics2":
        processor, model = utils.load_idefics2()
        extract_fn = idefics2_extract
    elif model_name == "llava":
        processor, model = load_llava()
        extract_fn = llava_extract
    elif model_name == "qwen":
        from qwen_vl_utils import process_vision_info
        processor, model = utils.load_qwenvl()
        processor.tokenizer.padding_side  = 'left'
        extract_fn = qwen_extract
    else:
        print("not implemented")
    
    ## init dataloaders  
    if dataset_name == "seedbench":
        # init seedbench dataset
        seedbench_dataset = SeedBenchDataset(ann_path, image_dir)
        # data loader
        data_loader = torch.utils.data.DataLoader(seedbench_dataset, batch_size=args.batch_size, shuffle=False)
    elif dataset_name == "foodieqa":
        foodieqa_dataset = FoodieQADataset(ann_path, image_dir)
        data_loader = torch.utils.data.DataLoader(foodieqa_dataset, batch_size=args.batch_size, shuffle=False)
    elif dataset_name == "vizwizgvqa":
        vizwizgvqa_dataset = VizWizGVQA(ann_path, image_dir)
        data_loader = torch.utils.data.DataLoader(vizwizgvqa_dataset, batch_size=args.batch_size, shuffle=False)
    elif dataset_name == "cub":
        cub_dataset = CUBDataset(image_dir)
        data_loader = torch.utils.data.DataLoader(cub_dataset, batch_size=args.batch_size, shuffle=False)
    elif dataset_name == "flickr30k" or dataset_name == "coco_test":
        flickr30k_dataset = Flickr30k(ann_path, image_dir)
        data_loader = torch.utils.data.DataLoader(flickr30k_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        print("Not implemented")
        
        
    pre_projection_store = FlatFileVectorBlockStore(args.out_dir, max_cache_size=100, prefix="pre")
    post_projection_store = FlatFileVectorBlockStore(args.out_dir, max_cache_size=100, prefix="post")
    
    batch_size = args.batch_size
    block2imageid = {}
    
    for i, batch in enumerate(tqdm(data_loader)):
        if model_name in ["llava", "idefics2", "qwen"]:
            emb_pre, emb_post = extract_fn(batch, processor, model)
            for idx in range(min(batch_size, len(batch["block_id"]))):
                block_id = batch["block_id"][idx]
                
                if args.avg_pool: # average pooling
                    pre_projection_store.store(block_id, torch.mean(emb_pre[idx], dim=0))
                    post_projection_store.store(block_id, torch.mean(emb_post[idx], dim=0))
                else:
                    pre_projection_store.store(block_id, emb_pre[idx])
                    post_projection_store.store(block_id, emb_post[idx])
                
                if dataset_name in ["foodieqa", "vizwizgvqa"]: # there is no image_id in foodieqa
                    block2imageid[block_id] = batch["image_file"][idx]
                else:
                    block2imageid[block_id] = batch["image_id"][idx] 
        else:
            print("Not implemented")

    pickle.dump(block2imageid, open(os.path.join(out_dir, f"{model_name}_block2imgid.pkl"), "wb"))
    print("Saved image ids to %s"% os.path.join(out_dir, f"{model_name}_block2imgid.pkl"))