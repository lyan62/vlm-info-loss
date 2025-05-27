import torch 
from torch.utils.data import Dataset, DataLoader
from vector_dataset import *
import faiss

from argparse import ArgumentParser
import os
import numpy
import yaml
import json


class CUBTestDataset(Dataset):
    def __init__(self, block_store: VectorBlockStore, test_blockids:dict, use_avg=False) -> None:
        """
        A dataset that returns `num_pairs` pairs of embeddings from the two block stores.
        """
        self.block_store = block_store
        self.use_avg = use_avg
        self.test_blockids = list(test_blockids.keys())
        self.bid2label = test_blockids

    def __len__(self) -> int:
        return len(self.test_blockids)

    def __getitem__(self, index: int) -> PairedEmbeddingsBatch:
        block_id = self.test_blockids[index]
        block = self.block_store.load(block_id)
        
        if self.use_avg:
            block = torch.mean(block, dim=0)
    
        return dict(
            embeddings=block,
            block_ids=block_id,
            label=self.bid2label[block_id]
        )
        

class EmbeddingsVectorBlockDataset(Dataset):
    def __init__(self, block_store: VectorBlockStore, use_avg=False) -> None:
        """
        A dataset that returns `num_pairs` pairs of embeddings from the two block stores.
        """
        self.block_store = block_store
        self.use_avg = use_avg

    def __len__(self) -> int:
        return len(self.block_store.block_ids)

    def __getitem__(self, index: int) -> PairedEmbeddingsBatch:
        block_id = self.block_store.block_ids[index]
        block = self.block_store.load(block_id)
        
        if self.use_avg:
            block = torch.mean(block, dim=0)
    
        return dict(
            embeddings=block,
            block_ids=block_id,
        )
        
        
def build_index(vector_dataset, vector_dataloader, save_dir="output", save_prefix="idefics2_test_pre", method="L2"):
    # Configuration
    embedding_dim = vector_dataset.__getitem__(0)["embeddings"].size(0)  # dimension of your embeddings
    os.makedirs(save_dir, exist_ok=True)
    index_file = os.path.join(save_dir, f"{save_prefix}.index")

    # For L2 distance:
    if method == "L2":
        index = faiss.IndexFlatL2(embedding_dim)
    else:
        # For cosine similarity (use IndexFlatIP with normalized vectors):
        index = faiss.IndexFlatIP(embedding_dim)
        
    for batch in vector_dataloader:
        vectors = batch["embeddings"].numpy().astype('float32')
        if method != "L2":
            faiss.normalize_L2(vectors)
        # Add to index
        index.add(vectors)

    # Write index to disk
    faiss.write_index(index, index_file)
    print(f"Index saved to {index_file} with {index.ntotal} embeddings")
    return index

def get_knn_search(data_loader, index, method, topk=100):
    distances = []
    indexes = []
    for idx, batch in enumerate(data_loader):
        query = batch["embeddings"].numpy().astype('float32')
        if method != "L2":
            faiss.normalize_L2(query)
        D, I = index.search(query, topk)  # search for the 5 nearest neighbors
        distances.append(D)
        indexes.append(I)
    # concat outputs
    distances = np.concatenate(distances, axis=0)
    indexes = np.concatenate(indexes, axis=0)
    return distances, indexes

def get_overlap(pre_ids, post_ids, topk=100):
    overlap = []
    for idx in range(pre_ids.shape[0]):
        overlap_cnt = np.intersect1d(pre_ids[idx, :topk], post_ids[idx, :topk])
        overlap.append(len(overlap_cnt)/topk)
    return overlap


def eval_cub(test_blockids, args):
    ## eval on pre embed dataset first
    pre_embed_store = FlatFileVectorBlockStore(args.test_vector_folder, max_cache_size=100, prefix="pre")
    pre_test_dataset = CUBTestDataset(pre_embed_store, test_blockids, use_avg=args.use_avg)
    
    pre_test_dataloader = DataLoader(
        pre_test_dataset,
        batch_size=100,
        shuffle=False
    )
    print("pre-vector embed size:", pre_test_dataset.__getitem__(0)["embeddings"].size(0))
    
    pre_index = build_index(pre_test_dataset, pre_test_dataloader, 
                            save_dir=args.index_save_dir, save_prefix="test_pre", method=args.method)
    
    pre_dist, pre_ids = get_knn_search(pre_test_dataloader, pre_index, args.method, topk=6) # top 6 for CUB
    
    np.save(os.path.join(args.index_save_dir, f"pre_dist_{args.topk}.npy"), pre_dist)
    np.save(os.path.join(args.index_save_dir, f"pre_ids_{args.topk}.npy"), pre_ids)
    print("pre search done, results saved to ", args.index_save_dir)
    
    post_embed_store = FlatFileVectorBlockStore(args.test_vector_folder, max_cache_size=100, prefix="post")
    post_test_dataset = CUBTestDataset(post_embed_store, test_blockids, use_avg=args.use_avg)
    
    post_test_dataloader = DataLoader(
        post_test_dataset,
        batch_size=50,
        shuffle=False
    )
    
    print("post-vector embed size:", post_test_dataset.__getitem__(0)["embeddings"].size(0))
    post_index = build_index(post_test_dataset, post_test_dataloader,
                             save_dir=args.index_save_dir, save_prefix="test_post", method=args.method)
    post_dist, post_ids = get_knn_search(post_test_dataloader, post_index, args.method, topk=6)
    np.save(os.path.join(args.index_save_dir, f"post_dist_{args.topk}.npy"), post_dist)
    np.save(os.path.join(args.index_save_dir, f"post_ids_{args.topk}.npy"), post_ids)
    print("post search done, results saved to ", args.index_save_dir)

    
        
if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--test_vector_folder", type=str, required=True)
    arg_parser.add_argument("--index_save_dir", type=str, default="/data/wenyan/output/knn_search_vqav2_val3000")
    arg_parser.add_argument("--topk", type=int, default=100)
    arg_parser.add_argument("--use_avg", action="store_true", default=False, help="use average embeddings when vectors are of size seq x dim")
    arg_parser.add_argument("--method", type=str, default="L2", help="distance metric for faiss index")
    arg_parser.add_argument("--eval", type=str, default="overlap", help="evaluation method")
    
    args = arg_parser.parse_args()
    
    if not os.path.exists(args.index_save_dir):
        os.makedirs(args.index_save_dir)
        
    if "cub" in args.test_vector_folder and args.eval == "retrieval":
        with open("/ceph/hpc/data/d2024d05-018-users/wenyan/code/VisU/CUB_200_2011/test_blockids.json", "r") as f:
            test_blockids = json.load(f)
        
        eval_cub(test_blockids, args)
    
    else: # neighborhood search
        print("load vector dataset")
        pre_embed_store = FlatFileVectorBlockStore(args.test_vector_folder, max_cache_size=100, prefix="pre")
        post_embed_store = FlatFileVectorBlockStore(args.test_vector_folder, max_cache_size=100, prefix="post")  
        
        # build vector dataset and dataloader, default is to use average embeddings
        pre_vector_dataset = EmbeddingsVectorBlockDataset(pre_embed_store, use_avg=args.use_avg)
        post_vector_dataset = EmbeddingsVectorBlockDataset(post_embed_store, use_avg=args.use_avg)
        
        print(pre_vector_dataset.__len__(), post_vector_dataset.__len__())
        
        pre_vector_dataloader = DataLoader(
                pre_vector_dataset,
                batch_size=100,
                shuffle=False
            )

        post_vector_dataloader = DataLoader(
                    post_vector_dataset,
                    batch_size=50,
                    shuffle=False
                )
        
        # build index
        # print("pre-vector embed size:", pre_vector_dataset.__getitem__(0)["embeddings"].size(0))
        # print("post-vector embed size:", post_vector_dataset.__getitem__(0)["embeddings"].size(0))
        
        pre_index = build_index(pre_vector_dataset, pre_vector_dataloader, 
                                save_dir=args.index_save_dir, save_prefix="test_pre")
        
        post_index = build_index(post_vector_dataset, post_vector_dataloader,
                                save_dir=args.index_save_dir, save_prefix="test_post")
        
        # search knn
        pre_dist, pre_ids = get_knn_search(pre_vector_dataloader, pre_index, args.method, topk=args.topk)
        post_dist, post_ids = get_knn_search(post_vector_dataloader, post_index, args.method, topk=args.topk)
        
        # save knn search results
        np.save(os.path.join(args.index_save_dir, f"pre_dist_{args.topk}.npy"), pre_dist)
        np.save(os.path.join(args.index_save_dir, f"pre_ids_{args.topk}.npy"), pre_ids)
        np.save(os.path.join(args.index_save_dir, f"post_dist_{args.topk}.npy"), post_dist)
        np.save(os.path.join(args.index_save_dir, f"post_ids_{args.topk}.npy"), post_ids)
        
        print("calculate overlap ratio")
        if args.topk == 100:
            overlap_top100 = get_overlap(pre_ids, post_ids, topk=100)
        if args.topk >= 50:
            overlap_top50 = get_overlap(pre_ids, post_ids, topk=50)
        if args.topk >= 10:
            overlap_top10 = get_overlap(pre_ids, post_ids, topk=10)
        
        # save overlap ratio
        with open(os.path.join(args.index_save_dir, f"overlap_ratio_{args.topk}.json"), "w") as f:
            json.dump({"top100": overlap_top100, "top50": overlap_top50, "top10": overlap_top10}, f, ensure_ascii=False)
        print("overlap ratio saved")
        print("mean overlap ratio top100:", np.mean(overlap_top100))
        print("mean overlap ratio top50:", np.mean(overlap_top50))
        print("mean overlap ratio top10:", np.mean(overlap_top10))
        