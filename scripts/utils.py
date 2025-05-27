import os
import torch
import numpy as np
import json

def load_idefics2():
    from transformers import AutoProcessor, AutoModelForVision2Seq
    model_name = "HuggingFaceM4/idefics2-8b"
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=os.environ["HF_HOME"])
    model = AutoModelForVision2Seq.from_pretrained(
        model_name, cache_dir=os.environ["HF_HOME"], device_map="auto", torch_dtype=torch.float16)
    processor.image_processor.size = {"longest_edge":336, "shortest_edge":336}
    processor.image_processor.do_image_splitting = False
    return processor, model

def load_llava():
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    model_id = "llava-hf/llava-1.5-7b-hf"

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=os.environ["HF_HOME"])
    model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16,
                                                        cache_dir=os.environ["HF_HOME"])

    return processor, model

def load_qwenvl():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2", 
        device_map="auto", cache_dir=os.environ["HF_HOME"]
    )
    min_pixels = 336*336 #256*28*28
    max_pixels = 336*336 #1280*28*28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", 
                                            min_pixels=min_pixels, max_pixels=max_pixels,
                                            device_map="auto", cache_dir=os.environ["HF_HOME"])
    processor.tokenizer.padding_side = "left"
    return processor, model
    

def get_model_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data