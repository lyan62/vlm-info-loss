import torch
import json
from transformers.image_utils import load_image
import os
from argparse import ArgumentParser
from huggingface_hub import login

from torch.utils.data import Dataset
import utils
import yaml
from tqdm import tqdm
from torchvision import transforms

# Replace "your_token_here" with your Hugging Face token
login("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")

class Evaluator:
    def __init__(self, dataset, model_name, dataset_config):
        self.dataset = dataset
        self.model_name = model_name
        self.dataset_config = dataset_config
        
        self.load_model()
    
    def load_model(self):
        if self.model_name == "llava":
            self.processor, self.model = utils.load_llava()
            print("loaded llava")
        elif self.model_name == "idefics2":
            self.processor, self.model = utils.load_idefics2()
            print("loaded idefics2")
        elif self.model_name == "qwen":
            self.processor, self.model = utils.load_qwenvl()
        else:
            raise ValueError("Invalid model name")
    
    def build_dataset(self):
        if self.dataset == "seedbench":
            dataset = SeedBenchQADataset(self.dataset_config["ann_path"], self.dataset_config["image_dir"])
        elif self.dataset == "foodieqa":
            dataset = FoodieSIVQA(self.dataset_config["ann_path"], self.dataset_config["image_dir"])
        else:
            raise ValueError("Invalid dataset name")
        return dataset
    
    def get_loader(self, dataset, batch_size=4, num_workers=1):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        return loader
    
    def get_llava_mcvqa_prompt(self, question, options):
        """
        <question>
        A. <option_1>
        B. <option_2>
        C. <option_3>
        D. <option_4>
        Answer with the option's letter from the given choices directly.
        """  
        template = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{question}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nAnswer with the option's letter from the given choices directly.\n"}
                    ]
            }
        ]
        template = self.processor.apply_chat_template(template, add_generation_prompt=True)
        return template
    
    def get_qwen_mcvqa_prompt(self, question, options, image_path):
        """
        <question>
        A. <option_1>
        B. <option_2>
        C. <option_3>
        D. <option_4>
        Answer with the option's letter from the given choices directly.
        """  
        resize_transform = transforms.Resize((336, 336))
        template = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": resize_transform(load_image(image_path))},
                    {"type": "text", "text": f"{question}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nAnswer with the option's letter from the given choices directly.\n"}
                    ]
            }
        ]
        return template
    
    def get_idefics2_mcvqa_prompt(self, question, options):
        prompt = f"<image>{question}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nAnswer with the option's letter from the given choices directly. Answer: "
        return prompt
    
    @staticmethod
    def collate_fn(batch):
        return {
            "image_path": [item["image_path"] for item in batch],
            "question": [item["question"] for item in batch],
            "question_id": [item["question_id"] for item in batch],
            "image_id": [item["image_id"] for item in batch],
            "options": [item["options"] for item in batch]
        }
        
    def get_option_probs(self, generate_ids):
        logits = generate_ids.scores  # logits for all generated tokens

        # Decode the generated tokens back to text
        tokenizer = self.processor.tokenizer 
        generated_sequences = generate_ids.sequences
        responses = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # Now handle probabilities for the options A, B, C, D
        option_tokens = [tokenizer.encode(option, add_special_tokens=False)[0] for option in ['A', 'B', 'C', 'D']]
        option_probs = []

        # For each question, get the log probabilities for only options A, B, C, D

        for log_prob in logits[0]:  # log_prob corresponds to the logits for each generated token
            # Extract log probabilities for the four options (A, B, C, D)
            option_log_probs = [log_prob[option_token].item() for option_token in option_tokens]

            # Convert log probabilities to probabilities (apply softmax to these four options)
            max_log_prob = max(option_log_probs)  # For numerical stability (avoid overflow in exp)
            log_probs_exp = [torch.exp(torch.tensor(log_prob - max_log_prob)) for log_prob in option_log_probs]  # Stabilize exp
            total_prob = sum(log_probs_exp)  # Sum of exponentiated log probabilities
            
            # Normalize to get actual probabilities
            option_probabilities = [log_prob_exp / total_prob for log_prob_exp in log_probs_exp]
            option_probs.append(option_probabilities)
            
        return responses, option_probs
    
    def evaluate_batch(self, batch, args):
        if self.model_name == "llava":
            images = [load_image(img_path) for img_path in batch['image_path']]
            prompts = [self.get_llava_mcvqa_prompt(q, opt) for q, opt in zip(batch['question'], batch['options'])]
        elif self.model_name == "idefics2":
            images = [[load_image(img_path)] for img_path in batch['image_path']]
            prompts = [self.get_llava_mcvqa_prompt(q, opt) for q, opt in zip(batch['question'], batch['options'])]
        elif self.model_name == "qwen":
            prompts = [self.get_qwen_mcvqa_prompt(q, opt, img_path) for q, opt, img_path in zip(batch['question'], batch['options'], batch['image_path'])]
            image_inputs, video_inputs = process_vision_info(prompts) 
            prompts = [self.processor.apply_chat_template(p, add_generation_prompt=True, tokenize=False) for p in prompts]
            # prompts = [self.get_idefics2_mcvqa_prompt(q, opt) for q, opt in zip(batch['question'], batch['options'])]
        else:
            raise ValueError("Invalid model name")
        
        
        if self.model_name in ["llava", "idefics2"]:
            inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt").to(self.model.device, torch.float16)
        elif self.model_name == "qwen":
            inputs = self.processor(images=image_inputs, videos=video_inputs, text=prompts, padding=True, return_tensors="pt").to(self.model.device, torch.float16)
        else:
            raise ValueError("Invalid model name")
            
        if args.get_option_probs:
            with torch.no_grad():
                generate_ids = self.model.generate(**inputs, max_new_tokens=5, 
                                                   min_length=5, 
                                                   pad_token_id=self.processor.tokenizer.pad_token_id,
                                                   output_scores=True, 
                                                   return_dict_in_generate=True)
                responses, option_probs = self.get_option_probs(generate_ids)
            return responses, option_probs
        else:
            with torch.no_grad():
                generate_ids = self.model.generate(**inputs, max_new_tokens=5, min_length=5)  # Ensure the model generates a minimum number of tokens)
                generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                responses = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return responses
    
    def evaluate(self, out_dir, args):
        with open(os.path.join(out_dir, f"{self.model_name}_{self.dataset}_pred_val.jsonl"), "w") as output_file:
            dataset = self.build_dataset()
            loader = self.get_loader(dataset, batch_size=args.batch_size)
            for idx, batch in tqdm(enumerate(loader), total=len(loader)):
                if args.get_option_probs:
                    responses, option_probs = self.evaluate_batch(batch, args)
                    # print("len(responses):", len(responses))
                    for i, (response, probs) in enumerate(zip(responses, option_probs)):
                        if type(probs) == list:
                            probs = [str(p.item()) for p in probs]
                        result = {
                            'question': batch['question'][i],
                            'question_id': batch['question_id'][i],
                            'image_id': batch['image_id'][i],
                            'response': response,
                            'option_probs': "|".join(probs)
                        }
                        output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                else:
                    responses = self.evaluate_batch(batch, args)
                    for i, response in enumerate(responses):
                        result = {
                            'question': batch['question'][i],
                            'question_id': batch['question_id'][i],
                            'image_id': batch['image_id'][i],
                            'response': response
                        }
                        output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

class SeedBenchQADataset(Dataset):
    def __init__(self, ann_path, image_dir, transform=None):
        self.transform = transform
        self.image_dir = image_dir
        self.data = utils.read_json(ann_path)
        self.questions = self._filter_questions()
        
        self.qid2ann = {ann['question_id']: ann for ann in self.questions}
        
    def __len__(self): 
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        qid = question['question_id']
        img_id = question['data_id']
        
        # Get validation image path
        img_path = os.path.join(self.image_dir, img_id)
        choices = [question['choice_a'], question['choice_b'], question['choice_c'], question['choice_d']]
        
        return {
            "image_path": img_path,
            "question": question['question'],
            "question_id": qid,
            "image_id": img_id,
            "options": choices
        }
        
    def _filter_questions(self):
        questions = [q for q in self.data["questions"] if 1 <= q["question_type_id"] <= 9]
        return questions
    
class FoodieSIVQA(Dataset):
    def __init__(self, ann_path, image_dir, transform=None):
        self.transform = transform
        self.image_dir = image_dir
        self.data = utils.read_json(ann_path)
        
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx):
        q = self.data[idx]
        question = q["question_en"]
        qid = q['question_id']
        img_file = q["food_meta"]["food_file"]
        food_meta_id = q["food_meta"]["id"]
        img_id = f"{qid}-{food_meta_id}"
        
        # Get validation image path
        img_path = os.path.join(self.image_dir, img_file)
        choices = q["choices_en"]
        
        return {
            "image_path": img_path,
            "question": question,
            "question_id": qid,
            "image_id": img_id,
            "options": choices
        }
    


if __name__ == "__main__": 
    argparser = ArgumentParser()
    argparser.add_argument("--data_config", type=str, default="/ceph/hpc/data/d2024d05-018-users/wenyan/code/VisU/datasets.yaml")
    argparser.add_argument("--image_dir", type=str, default="/ceph/hpc/data/d2024d05-018-users/wenyan/data/VisU/VQAv2/val2014")
    argparser.add_argument("--out_dir", type=str, default="/ceph/hpc/data/d2024d05-018-users/wenyan/data/VisU/eval_vqav2_val")
    argparser.add_argument("--model", choices=["llava", "idefics2", "qwen"], default="llava")
    argparser.add_argument("--dataset", choices=["seedbench", "foodieqa"], default="seedbench")
    argparser.add_argument("--batch_size", type=int, default=4)
    argparser.add_argument("--get_option_probs", action="store_true", default=False)
    
    args = argparser.parse_args()
    
    with open(args.data_config, "r") as file:
        config = yaml.safe_load(file)
    
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    model_name = args.model
    dataset_name = args.dataset
    # get dataset config
    if dataset_name in config["datasets"]:
        dataset_config = config["datasets"][dataset_name]
        ann_path = dataset_config["ann_path"]
        image_dir = dataset_config["image_dir"]
        print(f"Dataset: {dataset_name}")
        print(f"Annotation Path: {ann_path}")
        print(f"Image Directory: {image_dir}")
    else:
        print(f"Error: Dataset '{dataset_name}' not found in the configuration file.")
    
    if args.model == "qwen":
        from qwen_vl_utils import process_vision_info
    evaluator = Evaluator(dataset_name, model_name, dataset_config)
    evaluator.evaluate(out_dir, args)
            
        

        

