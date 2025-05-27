import argparse
import time
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Any, TypedDict, TypeVar

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset

T = TypeVar("T")


class PairedEmbeddingsBatch(TypedDict):
    embeddings: list[tuple[torch.Tensor, torch.Tensor]]
    block_ids: list[int]


def fixed_cache(maxsize: int = None) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
    cache = {}

    def wrapper(func: T) -> T:
        def cached_func(*args, **kwargs) -> Any:
            key = (args, tuple(kwargs.items()))

            if key in cache:
                return cache[key]

            result = func(*args, **kwargs)

            if key not in cache and (maxsize is None or len(cache) < maxsize):
                cache[key] = result

            return result

        return cached_func

    return wrapper


class RandomBlockCollator:
    def __init__(self, num_pairs_per_block: int) -> None:
        self.num_pairs_per_block = num_pairs_per_block

    def __call__(self, batch: list[PairedEmbeddingsBatch]) -> PairedEmbeddingsBatch:
        joined_embeddings = [(x, y) for item in batch for x, y in item["embeddings"]]
        block_ids = [x for item in batch for x in item["block_ids"]]
        # return joined_embeddings

        return dict(embeddings=joined_embeddings, block_ids=block_ids)


class VectorBlockStore:
    def load(self, block_id: int) -> torch.Tensor:
        raise NotImplementedError

    def store(self, block_id: int, tensor: torch.Tensor) -> None:
        raise NotImplementedError

    def delete(self, block_id: int) -> None:
        raise NotImplementedError

    @property
    def block_ids(self) -> list[int]:
        raise NotImplementedError


class FlatFileVectorBlockStore(VectorBlockStore):
    def __init__(self, path: str | Path, max_cache_size: int = 50, prefix: str = "") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.load = fixed_cache(maxsize=max_cache_size)(self.load)  # use a fixed cache for better random access
        self.prefix = f"{prefix}-" if prefix else ""

    def load(self, block_id: int) -> torch.Tensor:
        """Loads a block from the store. The block should be a 2D tensor with shape (num_embeddings, embedding_dim)."""
        return load_file(self.path / f"{self.prefix}block-{block_id}.safetensors")["block"]

    def store(self, block_id: int, tensor: torch.Tensor) -> None:
        """Stores a block in the store. The block should be a 2D tensor with shape (num_embeddings, embedding_dim)."""
        save_file(dict(block=tensor), self.path / f"{self.prefix}block-{block_id}.safetensors")

    def delete(self, block_id: int) -> None:
        """Deletes a block from the store."""
        (self.path / f"{self.prefix}block-{block_id}.safetensors").unlink()

    @cached_property
    def block_ids(self) -> list[int]:
        """Returns the list of block IDs in the store."""
        return sorted([int(path.stem.split("-")[-1]) for path in self.path.glob(f"{self.prefix}block-*.safetensors")])


class PairedEmbeddingsVectorBlockDataset(Dataset):
    def __init__(self, block_store1: VectorBlockStore, block_store2: VectorBlockStore, num_pairs: int) -> None:
        """
        A dataset that returns `num_pairs` pairs of embeddings from the two block stores.
        """
        assert len(block_store1.block_ids) == len(block_store2.block_ids), "must be same number of blocks"
        assert set(block_store1.block_ids) == set(block_store2.block_ids), "must be same blocks"
        self.block_store1 = block_store1
        self.block_store2 = block_store2
        self.num_pairs = num_pairs

    def __len__(self) -> int:
        return len(self.block_store1.block_ids)

    def __getitem__(self, index: int) -> PairedEmbeddingsBatch:
        block_id = self.block_store1.block_ids[index]
        block1 = self.block_store1.load(block_id)
        block2 = self.block_store2.load(block_id)
        rand_indices = np.random.permutation(len(block1))[: self.num_pairs]

        return dict(
            embeddings=list(zip(block1[rand_indices], block2[rand_indices], strict=False)),
            block_ids=[block_id] * self.num_pairs,
        )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["store", "load"], required=True)
    parser.add_argument("--folder", type=str, default="output")
    parser.add_argument("--num-blocks", type=int, default=150)
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--num-obs-per-block", type=int, default=300)
    parser.add_argument("--block-embedding-dim", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    store1 = FlatFileVectorBlockStore(args.folder, max_cache_size=100, prefix="store1")
    store2 = FlatFileVectorBlockStore(args.folder, max_cache_size=100, prefix="store2")

    match args.action:
        case "store":
            for block_id in range(args.num_blocks):
                block1 = torch.randn(args.num_obs_per_block, args.block_embedding_dim)
                block2 = torch.randn(args.num_obs_per_block, args.block_embedding_dim)
                store1.store(block_id, block1)
                store2.store(block_id, block2)
        case "load":
            a = time.time()

            for block_id in range(args.num_blocks):
                block1 = store1.load(block_id)
                block2 = store2.load(block_id)

            b = time.time()
            print(f"uncached time: {b - a}")
            a = time.time()

            for block_id in range(args.num_blocks):
                block1 = store1.load(block_id)
                block2 = store2.load(block_id)

            b = time.time()
            print(f"cached time: {b - a}")

            train_dataloader = DataLoader(
                PairedEmbeddingsVectorBlockDataset(store1, store2, args.num_pairs),
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=RandomBlockCollator(args.num_pairs),
            )

            for batch in train_dataloader:
                # print(batch[0])
                print(batch["embeddings"][0][0].shape, batch["embeddings"][0][1].shape, batch["block_ids"])
                break


if __name__ == "__main__":
    main()
