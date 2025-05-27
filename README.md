# Information Loss for Vision-Language Models
## Preparation
Download the dataset such as Seedbench and COCO from their official website, and setup annotation and data path in `dataset.yaml` (example provided).
Note that the extracted visual embeddings could be quite large, for example for LLAVA it could takes around 1T for COCO datset. Make sure you have enough storage before extracting the embeddings.
## Extract embeddings

Extract embeddings for the entire image set for the dataset (SEEDBENCH, foodieQA, etc.)

```
export MODEL=llava
export DATA=seedbench
python scripts/get_vector_blocks_full_dataset.py --data_config datasets.yaml \
    --model $MODEL --dataset $DATA --out_dir embed/${MODEL}_vector_blocks_${DATA}
```

## FAISS KNN search with pre/post embeddings
```
export DATA=cub
export MODEL=llava
export METHOD="L2"
python scripts/faiss_knn_search.py --test_vector_folder ${MODEL}_vector_blocks_${DATA} \
    --index_save_dir embed/${MODEL}_vector_blocks_${DATA}/knn_index_${METHOD} \
    --use_avg --method $METHOD
```

## Reconstruct embeddings
### Train reconstruction model
```
export MODEL=idefics2
export DATA=coco
export BS=64
export EPOCHS=30
export LR=1e-4
export TYPE=mlp
export HID_DIM=3072
export SEED=42
export LAYERS=16
export OUT_PATH=${MODEL}_${DATA}_bs${BS}_lr${LR}_epochs${EPOCHS}_hiddim${HID_DIM}_layers${LAYERS}_seed${SEED}_normalize

OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/reconstruct_embeddings_parallel.py --embed_model $MODEL \
    --vector_folder embed/${MODEL}_vector_blocks_${DATA} \
    --out_dir output/reconstruct_parallel/$OUT_PATH \
    --batch_size $BS --lr $LR --epochs $EPOCHS --hidden_size $HID_DIM --num_layers $LAYERS --model_type $TYPE --seed $SEED \
    --normalize 
```

### Eval reconstruction model
```
python scripts/reconstruct_embeddings.py --embed_model $MODEL \
    --test_vector_folder embed/${MODEL}_vector_blocks_${DATA} \
    --model_dir $MODEL_DIR \
    --out_dir output/embed_reconstruction_res/pred/$OUT_PATH \
    --hidden_size $HID_DIM --model_type $TYPE --num_layers $NUM_LAYERS --eval --normalize
```

## Eval Captioning
```
python scripts/eval_caption.py --data_config datasets.yaml \
    --dataset $DATA \
    --model $MODEL --out_dir /output/caption/${MODEL}_${DATA}_res
```

## Eval Multiple-choice question answering
```
python scripts/eval_multiple_choice.py --data_config datasets.yaml \
    --model $MODEL --dataset $DATA \
    --out_dir output/mcvqa/${MODEL}_${DATA}_res \
    --get_option_probs --batch_size 2
```

## Visualization
We provide the visualization of reconstructed embeddings in notebook `scripts/vis_vizwizgvqa.ipynb`