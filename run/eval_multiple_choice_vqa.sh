
export HF_DATASETS_CACHE=<cache_path>
export HF_HOME=<hf_path>

### get embeddings for dataset
export MODEL=qwen
export DATA=coco_test
python scripts/eval_multiple_choice.py --data_config datasets.yaml \
    --model $MODEL --dataset $DATA \
    --out_dir ${MODEL}_${DATA}_res \
    --get_option_probs --batch_size 2


## eval captioning
python scripts/eval_caption.py --data_config datasets.yaml \
    --dataset $DATA \
    --model $MODEL --out_dir ${MODEL}_${DATA}_res