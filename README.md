## Extract embeddings

Extract embeddings by specifying image indexs (applies to VQAv2 and COCO)
```
python scripts/get_vector_blocks.py --data_config /ceph/hpc/data/d2024d05-018-users/wenyan/code/VisU/datasets.yaml \
     --model $MODEL --dataset $DATA \
     --out_dir /ceph/hpc/data/d2024d05-018-users/wenyan/data/VisU/output/embed/${MODEL}_vector_blocks_${DATA} \
     --range "10000,20000"

```

Extract embeddings for the entire image set for the dataset (SEEDBENCH, foodieQA, etc.)

```
export MODEL=llava
export DATA=seedbench
python scripts/get_vector_blocks_full_dataset.py --data_config /ceph/hpc/data/d2024d05-018-users/wenyan/code/VisU/datasets.yaml \
    --model $MODEL --dataset $DATA --out_dir /ceph/hpc/data/d2024d05-018-users/wenyan/data/VisU/output/embed/${MODEL}_vector_blocks_${DATA}
```

## FAISS KNN search with pre/post embeddings
```
python scripts/faiss_knn_search.py \
    --test_vector_folder /ceph/hpc/data/d2024d05-018-users/wenyan/data/VisU/llava_vector_blocks/test \
    --index_save_dir /ceph/hpc/data/d2024d05-018-users/wenyan/data/VisU/llava_faiss_index/test 
```

## Reconstruct embeddings
### Train reconstruction model
```
python scripts/reconstruct_embeddings.py --embed_model idefics2 --test_vector_folder /data/wenyan/output/VisU/idefics2_vector_blocks/test \
    --out_dir /data/wenyan/output/VisU/embed_reconstruction_res/test \
    --model_dir /data/wenyan/output/VisU/embed_reconstruction_res > /data/wenyan/output/VisU/logs/idefics2_emb_reconstruct_eval.log 2>&1
```

### Eval reconstruction model
```
python scripts/reconstruct_embeddings.py --embed_model idefics2  --eval \
    --test_vector_folder /data/wenyan/output/VisU/idefics2_vector_blocks/test \
    --out_dir /data/wenyan/output/VisU/embed_reconstruction_res/test \
    --model_dir <path to model ckpt dir> > /data/wenyan/output/VisU/logs/idefics2_emb_reconstruct_eval.log 2>&1
```