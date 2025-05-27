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
import yaml
import pickle
from torchvision import transforms
import pdb

# Replace "your_token_here" with your Hugging Face token
login("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")


class COCOTrainDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir))
        print("Number of images: ", len(self.image_files))
        self.imgid2file = {int(img_file.strip(".jpg").lstrip("0")): img_file for img_file in self.image_files}
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        block_id = idx
        image_id = int(image_file.strip(".jpg").lstrip("0"))
        return {"image_file": image_file, "image_path": image_path, "block_id": block_id, "image_id": image_id}
    
    
def load_chameleon():
    from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
    processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf",
                                                   cache_dir=os.environ["HF_HOME"])
    model = ChameleonForConditionalGeneration.from_pretrained("leloy/Anole-7b-v0.1-hf", 
                                                              cache_dir=os.environ["HF_HOME"],
                                                              torch_dtype=torch.bfloat16, device_map="cuda")
    return processor, model

def get_llava_prompt(image_id, img2q, processor):
    if img2q is not None:
        text = img2q[image_id][0][0]
    else:
        text = "What do you see in this image?"
        
    template = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        },
    ]
    
    prompt = processor.apply_chat_template(template, add_generation_prompt=True)
    return prompt

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

def get_qwen_prompt(image_id, img2q, image_path):
    if img2q is not None:
        text = img2q[image_id][0][0]
    else:
        text = "What do you see in this image?"
        
    resize_transform = transforms.Resize((336, 336))
    template = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": resize_transform(load_image(image_path)),
                },
                {"type": "text", "text": text},
            ],
        }
    ]
    return template

def chameleon_extract(image_path, image_id, i, vqav2_val, processor, model):
    # extract embeddings from chameleon
    image = load_image(image_path)
    prompt = vqav2_val["questions"][i]["question"] + "<image>"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    
    with torch.no_grad():
        quant, loss, indices = model.model.vqmodel.encode(inputs['pixel_values'])
        bpe_tokens = model.model.get_image_tokens(inputs['pixel_values'])
        output = model.generate(**inputs, max_new_tokens=30)
        generate_ids = output[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    
    out_dict = {"image_id": image_id, 
                "quant": quant.float().cpu().detach().numpy(), 
                "loss": loss.float().cpu().detach().numpy(), 
                "indices": indices.int().cpu().detach().numpy(), 
                "bpe_tokens": bpe_tokens.float().cpu().detach().numpy(),
                "response": response}
    return out_dict

def get_inputs_vqav2(input_batch, img2q):
    images = [load_image(x[-1]) for x in input_batch]
    prompts = [get_llava_prompt(x[0], img2q) for x in input_batch]
    return images, prompts

def get_inputs_coco(input_batch, processor):
    images = [load_image(x["image_path"]) for x in input_batch]
    prompts = [get_llava_prompt(x["image_id"], None, processor) for x in input_batch]
    return images, prompts

def qwen_extract(input_batch, img2q, processor, model, dataset_name):
    if dataset_name == "vqav2":
        templates = [get_qwen_prompt(x[0], img2q, x[-1]) for x in input_batch]
        image_ids = [x[0] for x in input_batch]
    elif dataset_name == "coco":
        templates = [get_qwen_prompt(x["image_id"], None, x["image_path"]) for x in input_batch]
        image_ids = [x["image_id"] for x in input_batch]
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
    connector_hook.remove()
    
    hidden_dim_before = hidden_states_before[0].size(-1) #1280
    hidden_dim_after = hidden_states_after[0].size(-1) #3584
    
    pre_projection = hidden_states_before[0].view(len(input_batch), -1, hidden_dim_before).float().cpu().detach()
    post_projection = hidden_states_after[0].view(len(input_batch), -1, hidden_dim_after).float().cpu().detach()
    return image_ids, pre_projection, post_projection 
    
    
def llava_extract(input_batch, img2q, processor, model, dataset_name):
    # input_batch = [image_id, image_file, image_path]
    if dataset_name == "vqav2":
        images, prompts = get_inputs_vqav2(input_batch, img2q)
        image_ids = [x[0] for x in input_batch]
    elif dataset_name == "coco":
        images, prompts = get_inputs_coco(input_batch, processor)
        image_ids = [x["image_id"] for x in input_batch]
    else:
        print("Dataset not found")
    
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
        # generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        # response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    connector_hook.remove()
    pre_projection = hidden_states_before[0].float().cpu().detach() #.numpy()
    post_projection = hidden_states_after[0].float().cpu().detach() #.numpy()
    # print(pre_projection.shape, post_projection.shape)
    
    return image_ids, pre_projection, post_projection # 2, 576, 1024/4096
        

def get_llava_batch_helper(i, img2q_list, image_dir):
    image_id = img2q_list[i][0] # image_id
    image_file = "COCO_val2014_" + str(image_id).zfill(12) + ".jpg"
    image_path = os.path.join(image_dir, image_file)
    return [image_id, image_file, image_path]
    
def idefics2_extract(input_batch, img2q, processor, model, dataset_name):
    # image = load_image(image_path)
    # images = [image]
    # prompt = img2q[image_id][1][0] + "<image>"  # get the first question for the image
    # batched inputs
    if dataset_name == "vqav2":
        images = [[load_image(x[-1])] for x in input_batch]
        prompts = [img2q[x[0]][1][0] + "<image>" for x in input_batch]
    elif dataset_name == "coco":
        # images = [load_image(x["image_path"]) for x in input_batch]
        # prompts = ["What do you see in this image?" + "<image>" for x in input_batch]
        images, prompts = get_inputs_coco(input_batch, processor)
        image_ids = [x["image_id"] for x in input_batch]
    else:
        print("Dataset not found")
    # fix input images to be the same size
    inputs = processor(text=prompts, images=[[x] for x in images], padding=True, return_tensors="pt")
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
    if dataset_name == "vqav2":
        image_ids = [x[0] for x in input_batch]
    else:
        image_ids = [x["image_id"] for x in input_batch]
    pre_projection = hidden_states_before[0].float().cpu().detach() #.numpy()
    post_projection = hidden_states_after[0].float().cpu().detach() #.numpy()
    
    return image_ids, pre_projection, post_projection
        # return image_ids, pre_projection, post_projection, response # [], [2, 576, 1152], [2,576, 4096]


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

if __name__ == "__main__": 
    argparser = ArgumentParser()
    # argparser.add_argument("--ann_path", type=str, default="/data/wenyan/vqav2/v2_OpenEnded_mscoco_val2014_questions.json")
    # argparser.add_argument("--image_dir", type=str, default="/data/wenyan/vqav2/val2014")
    argparser.add_argument("--data_config", type=str, default="/ceph/hpc/data/d2024d05-018-users/wenyan/code/VisU/datasets.yaml")
    argparser.add_argument("--out_dir", type=str, default="/ceph/hpc/data/d2024d05-018-users/wenyan/data/llava_blocks")
    argparser.add_argument("--model", type=str, default="llava")
    argparser.add_argument("--save_format", type=str, default="np")
    argparser.add_argument("--range", type=str, default="10000,20000")
    argparser.add_argument("--avg_pool", action="store_true", default=False)
    argparser.add_argument("--dataset", choices=["vqav2", "coco"], default="vqav2")
    argparser.add_argument("--batch_size", type=int, default=16)

    
    args = argparser.parse_args()
    
    out_dir = args.out_dir
    model_name = args.model
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    ### load yaml file and dataset annotations
    with open(args.data_config, "r") as file:
        config = yaml.safe_load(file)
    
    dataset_name = args.dataset
    if dataset_name in config["datasets"]:
        dataset_config = config["datasets"][dataset_name]
        ann_path = dataset_config.get("ann_path", None)
        image_dir = dataset_config["image_dir"]
        print(f"Dataset: {dataset_name}")
        print(f"Annotation Path: {ann_path}")
        print(f"Image Directory: {image_dir}")
    else:
        print(f"Error: Dataset '{dataset_name}' not found in the configuration file.")
    

    if model_name == "idefics2":
        processor, model = utils.load_idefics2()
        processor.image_processor.size = {"longest_edge":336, "shortest_edge":336}
        processor.image_processor.do_image_splitting = False
        extract_fn = idefics2_extract
    elif model_name == "llava":
        processor, model = utils.load_llava()
        extract_fn = llava_extract
    elif model_name == "qwen":
        from qwen_vl_utils import process_vision_info
        processor, model = utils.load_qwenvl()
        extract_fn = qwen_extract
    else:
        print("Model not found")
    
    batch = []
    batch_size = args.batch_size
      
    # load vqav2 validation set
    if dataset_name == "vqav2":
        vqav2_val = utils.read_json(ann_path)
    
        # unique_image_ids = list(set([x["image_id"] for x in vqav2_val["questions"]]))
        
        # extract unique image_id
        img2q = defaultdict(list)
        for item in vqav2_val["questions"]:
            img2q[item["image_id"]].append((item["question"], item["question_id"]))
        img2q_list = list(img2q.items())
        print("Number of unique images: ", len(img2q_list))
    elif dataset_name == "coco":
        img2q_list = None
        img2q = None
        coco_dataset = COCOTrainDataset(image_dir)
    else:
        print("Dataset not found")
        
    
    
    block2imageid = {}    
    pre_projection_store = FlatFileVectorBlockStore(args.out_dir, max_cache_size=100, prefix="pre")
    post_projection_store = FlatFileVectorBlockStore(args.out_dir, max_cache_size=100, prefix="post")
    
    range_start, range_end = map(int, args.range.split(","))
    range_end = min(range_end, len(img2q_list) if img2q_list is not None else len(coco_dataset))
    
    for i in tqdm(range(range_start, range_end, batch_size)):
        # image_id = vqav2_val["questions"][i]["image_id"]
        # image_file = "COCO_val2014_" + str(vqav2_val['questions'][0]['image_id']).zfill(12) + ".jpg"
        if model_name in ["llava", "idefics2", "qwen"]:
            input_batch = []
            if dataset_name == "vqav2":
                for idx in range(batch_size):
                    input_batch.append(get_llava_batch_helper(i+idx, img2q_list, image_dir))    
            else:
                for idx in range(batch_size):
                    input_batch.append(coco_dataset.__getitem__(i+idx))   
            # extract embeddings
            image_ids, emb_pre, emb_post = extract_fn(input_batch, img2q, processor, model, dataset_name)
            # store results
            for idx in range(batch_size):
                if args.avg_pool:
                    pre_projection_store.store(i+idx, torch.mean(emb_pre[idx], axis=0)) ## save the vector blocks
                    post_projection_store.store(i+idx, torch.mean(emb_post[idx], axis=0))
                else:
                    pre_projection_store.store(i+idx, emb_pre[idx]) ## save the vector blocks
                    post_projection_store.store(i+idx, emb_post[idx]) 
                block2imageid[i+idx] = image_ids[idx]
                      
        else:
            print("not implemented")
            
    pickle.dump(block2imageid, open(os.path.join(out_dir, f"{model_name}_block2imgid_{range_start}_{range_end}.pkl"), "wb"))
    print("Saved image ids to %s"% os.path.join(out_dir, f"{model_name}_block2imgid_{range_start}_{range_end}.pkl"))
            