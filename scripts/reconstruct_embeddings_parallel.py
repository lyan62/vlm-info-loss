import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
import gc
from torch.cuda.amp import autocast, GradScaler
import os
from tqdm import tqdm 
from vector_dataset import *
import utils
from argparse import ArgumentParser
import random
import pdb
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F


class PairedVectorBlock(TypedDict):
    embeddings: tuple[torch.Tensor, torch.Tensor]
    block_id: int
    

class PairedVectorBlockDataset(Dataset):
    def __init__(self, block_store1: VectorBlockStore, block_store2: VectorBlockStore, normalize_vectors=False) -> None:
        """
        A dataset that returns `num_pairs` pairs of embeddings from the two block stores.
        """
        assert len(block_store1.block_ids) == len(block_store2.block_ids), "must be same number of blocks"
        assert set(block_store1.block_ids) == set(block_store2.block_ids), "must be same blocks"
        self.block_store1 = block_store1
        self.block_store2 = block_store2
        self.normalize_vectors = normalize_vectors
    def __len__(self) -> int:
        return len(self.block_store1.block_ids)

    def __getitem__(self, index: int) -> PairedVectorBlock:
        block_id = self.block_store1.block_ids[index]
        block1 = self.block_store1.load(block_id)
        block2 = self.block_store2.load(block_id)
        
        return dict(
            embeddings=(block1, block2),
            block_id=block_id
        )
        

class EmbeddingTransformer(nn.Module):
    def __init__(self, input_dim=4096, output_dim=1024, hidden_dim=2048, num_layers=3):
        super().__init__()
        hidden_dim = min(hidden_dim, max(input_dim, output_dim))
        
        if num_layers == 3:
            self.model = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),  # Changed from ReLU
                nn.Dropout(0.1),
                
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),  # Changed from ReLU
                nn.Dropout(0.1),
                
                nn.Linear(hidden_dim, output_dim)
            )
        elif num_layers > 3:
            layers = [
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),  # Changed from ReLU
                nn.Dropout(0.1),
                
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),  # Changed from ReLU
                nn.Dropout(0.1),    
            ]
            
            for _ in range(num_layers - 3):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),  # Changed from ReLU
                    nn.Dropout(0.1)
                ])
            
            layers.append(nn.Linear(hidden_dim, output_dim))
            
            self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class LargerEmbeddingTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 4096,
        output_dim: int = 1024,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        seq_length: int = 576
    ):
        """
        A Transformer-based model for embedding reconstruction.
        
        Args:
            input_dim (int): Dimensionality of each input token (default 4096).
            output_dim (int): Desired output dimensionality for each token (default 1024).
            hidden_dim (int): Internal hidden dimension for the Transformer (default 2048).
            num_layers (int): Number of Transformer encoder layers.
            num_heads (int): Number of attention heads in each Transformer encoder layer.
            dropout (float): Dropout rate.
            seq_length (int): Sequence length (number of tokens), here fixed to 576.
        """
        super().__init__()
        # Project input embeddings into a hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Learnable positional embeddings for the sequence tokens.
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, hidden_dim))
        
        # Create a stack of Transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            activation="gelu",
            batch_first=True  # since our input is (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_encoder.gradient_checkpointing = True
        
        # Project from hidden dimension down to the target output dimension.
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # (Optional) Final layer normalization for stabilization
        self.final_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
        Returns:
            torch.Tensor: Reconstructed embeddings of shape (batch_size, seq_length, output_dim)
        """
        # x: (batch, seq_length, input_dim)
        x = self.input_proj(x)  # (batch, seq_length, hidden_dim)
        x = x + self.pos_embedding  # add positional information
        
        # Process with Transformer encoder.
        x = self.transformer_encoder(x)  # (batch, seq_length, hidden_dim)
        
        x = self.output_proj(x)  # (batch, seq_length, output_dim)
        x = self.final_norm(x)   # optional normalization
        return x
    
class SeqEmbeddingTransformer(nn.Module):
    def __init__(self, input_dim=4096, input_seq_len=64, output_seq_len=576, 
                output_dim=1152, hidden_dim=2048, num_layers=8):
        super().__init__()
        
        # Enhanced input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Learnable positional embeddings
        self.pos_encoder = nn.Parameter(torch.randn(1, input_seq_len, hidden_dim))
        
        # Transformer with pre-norm and reduced depth
        self.sequence_processor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=24, #16
                dim_feedforward=hidden_dim*4,
                dropout=0.1,
                batch_first=True,
                norm_first=True  # Critical for stability
            ),
            num_layers=num_layers  # Reduced from 16
        )
        
        # Improved sequence length adjustment
        self.length_adjust = nn.Sequential(
            nn.Linear(input_seq_len, output_seq_len * 2),
            nn.GELU(),
            nn.Linear(output_seq_len * 2, output_seq_len)
        )
        
        # Enhanced output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, x):
        # Input processing
        x = self.input_projection(x)  # [B, L_in, H]
        x = x + self.pos_encoder  # Add positional information
        
        # Sequence processing
        x = self.sequence_processor(x)  # [B, L_in, H]
        
        # Length adjustment
        x = x.transpose(1, 2)  # [B, H, L_in]
        x = self.length_adjust(x)  # [B, H, L_out]
        x = x.transpose(1, 2)  # [B, L_out, H]
        
        # Final projection
        return self.output_projection(x)


def standardize_embeddings(embeddings: torch.Tensor, mean, std):
    """
    Standardize embeddings across a dataset (not per-sample!).
    Args:
        embeddings: Shape [N, D] (N samples, D dimensions)
        dim: Dimension to compute mean/std (0 for dataset-wide stats).
    """
    return (embeddings - mean) / (std + 1e-8)  # Avoid division by zero


def train_model(model, train_loader, val_loader, config, args, local_rank, world_size, device='cuda'):
    if local_rank != 0:
        wandb.init(mode="disabled")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Schedulers setup remains the same
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs * len(train_loader) - config.warmup_steps,
        eta_min=config.learning_rate * 0.01
    )
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / config.warmup_steps)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[config.warmup_steps]
    )
    
    scaler = GradScaler(enabled=config.mixed_precision)
    best_val_loss = float('inf')
    best_model = None
    patience = 10
    patience_counter = 0
    min_delta = 1e-3
    accumulation_steps = 4
    
    if args.normalize:
        print("calculate dataset mean and std")
        stats_dict = compute_dataset_stats(train_loader)
   

    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = torch.zeros(1).to(device)
        optimizer.zero_grad()

        # Only create progress bar on main process
        if local_rank == 0:
            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Training]", ncols=100)
        else:
            train_loader_tqdm = train_loader

        for batch_idx, batch in enumerate(train_loader_tqdm):
            if args.normalize:
                post_emb = standardize_embeddings(batch["embeddings"][1], stats_dict["mean_post"], stats_dict["std_post"]).to(device, non_blocking=True)
                pre_emb = standardize_embeddings(batch["embeddings"][0], stats_dict["mean_pre"], stats_dict["std_pre"]).to(device, non_blocking=True)
            else:
                post_emb = batch["embeddings"][1].to(device, non_blocking=True)
                pre_emb = batch["embeddings"][0].to(device, non_blocking=True)

            with autocast():
                output = model(post_emb)
                loss = criterion(output, pre_emb)
                # Average loss across GPUs for DDP
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    loss = loss.mean()
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            # Accumulate loss
            train_loss += loss.item() * accumulation_steps

            # Logging only on main process
            if local_rank == 0:
                if isinstance(train_loader_tqdm, tqdm):
                    train_loader_tqdm.set_postfix({"Batch Loss": loss.item() * accumulation_steps})
                if batch_idx % config.log_interval == 0:
                    wandb.log({
                        "batch_loss": loss.item() * accumulation_steps,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "epoch": epoch,
                        "batch": batch_idx
                    })

        # Synchronize and average training loss across GPUs
        if world_size > 1:
            dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
            train_loss = train_loss / world_size
        train_loss = train_loss.item() / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = torch.zeros(1).to(device)

        if local_rank == 0:
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Validation]", ncols=100)
        else:
            val_loader_tqdm = val_loader

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader_tqdm):
                if args.normalize:
                    post_emb = standardize_embeddings(batch["embeddings"][1], stats_dict["mean_post"], stats_dict["std_post"]).to(device, non_blocking=True)
                    pre_emb = standardize_embeddings(batch["embeddings"][0], stats_dict["mean_pre"], stats_dict["std_pre"]).to(device, non_blocking=True)
                else:
                    post_emb = batch["embeddings"][1].to(device, non_blocking=True)
                    pre_emb = batch["embeddings"][0].to(device, non_blocking=True)
                
                with autocast():
                    output = model(post_emb)
                    loss = criterion(output, pre_emb)
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        loss = loss.mean()
                
                val_loss += loss.item()

                if local_rank == 0 and isinstance(val_loader_tqdm, tqdm):
                    val_loader_tqdm.set_postfix({"Val Loss": loss.item()})

                del output, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Synchronize and average validation loss
            if world_size > 1:
                dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                val_loss = val_loss / world_size
            val_loss = val_loss.item() / len(val_loader)

        # Early stopping and logging only on main process
        if local_rank == 0:
            if val_loss < best_val_loss - min_delta:
                print(f"Validation loss decreased ({best_val_loss:.6f} --> {val_loss:.6f})")
                best_val_loss = val_loss
                # Save model state
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    best_model = model.module.state_dict()
                else:
                    best_model = model.state_dict()
                torch.save(best_model, os.path.join(args.out_dir, "best_model.pt"))
                wandb.save("best_model.pt")
                patience_counter = 0
                print("model saved to ", os.path.join(args.out_dir, "best_model.pt"))
            else:
                patience_counter += 1

            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "patience_counter": patience_counter
            })

            print(f'Epoch {epoch+1}/{config.num_epochs}:')
            print(f'Train Loss: {train_loss:.6f}')
            print(f'Val Loss: {val_loss:.6f}')
            print(f'Patience Counter: {patience_counter}/{patience}')
            print('-' * 30)

            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

        # Synchronize at end of epoch
        if world_size > 1:
            dist.barrier()

    return best_model


def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    total_loss = 0
    
    # Store detailed results for each sample
    sample_results = {
        'block_ids': [],
        'sample_losses': [],
        'feature_losses': [],  # Store per-feature losses
        'predictions': [],
        'targets': []
    }
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            post_emb = batch["embeddings"][1].to(device)
            pre_emb = batch["embeddings"][0].to(device)
            block_ids = batch["block_id"]
            
            with autocast():
                output = model(post_emb)
                # Calculate per-sample, per-feature loss
                losses = criterion(output, pre_emb)
                
                # Average over features dimension for sample-level loss
                sample_losses = losses.mean(dim=2) # average over feature dim, this is of seq-len length
                
                total_loss += sample_losses.sum().item()
                
                # Store detailed results
                sample_results['block_ids'].extend(block_ids)
                sample_results['sample_losses'].extend(sample_losses.cpu().numpy())
                sample_results['feature_losses'].extend(losses.cpu().numpy())
                sample_results['predictions'].extend(output.cpu().numpy())
                sample_results['targets'].extend(pre_emb.cpu().numpy())
                
            del output, losses
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(test_loader.dataset)
    
    # Convert lists to numpy arrays for easier processing
    sample_results['sample_losses'] = np.array(sample_results['sample_losses'])
    sample_results['feature_losses'] = np.array(sample_results['feature_losses'])
    sample_results['predictions'] = np.array(sample_results['predictions'])
    sample_results['targets'] = np.array(sample_results['targets'])
    
    # Calculate statistics
    stats = {
        'average_loss': avg_loss,
        'median_loss': np.median(sample_results['sample_losses']),
        'std_loss': np.std(sample_results['sample_losses']),
        'min_loss': np.min(sample_results['sample_losses']),
        'max_loss': np.max(sample_results['sample_losses']),
        'percentile_25': np.percentile(sample_results['sample_losses'], 25),
        'percentile_75': np.percentile(sample_results['sample_losses'], 75)
    }
    
    return stats, sample_results

def get_model_params(module):
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total_params, trainable_params


def setup(rank, world_size):
    # With torchrun, MASTER_ADDR and MASTER_PORT are already set in the environment
    # Initialize the process group using the provided env variables.
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
    

def compute_dataset_stats(dataloader):
    """
    Compute mean and std for `pre_emb` and `post_emb` independently, even if their sequence lengths differ.
    Args:
        dataloader: Yields batches with "embeddings" key containing tuples of (pre_emb, post_emb).
                    Shapes: pre_emb [B, Seq_len_pre, D], post_emb [B, Seq_len_post, D].
    Returns:
        (mean_pre, std_pre), (mean_post, std_post)
    """
    # Initialize accumulators for pre_emb and post_emb
    sum_pre, sum_sq_pre, num_samples_pre = None, None, 0
    sum_post, sum_sq_post, num_samples_post = None, None, 0

    for batch in tqdm(dataloader, total=len(dataloader), desc="Computing Dataset Stats"):
        # Extract pre_emb and post_emb from the batch
        pre_emb = batch["embeddings"][0]  # Shape [B, Seq_len_pre, D]
        post_emb = batch["embeddings"][1]  # Shape [B, Seq_len_post, D]

        # Flatten batch and sequence dimensions for pre_emb
        B_pre, Seq_len_pre, D_pre = pre_emb.shape
        pre_emb_flat = pre_emb.reshape(-1, D_pre)  # [B * Seq_len_pre, D]
        curr_num_pre = pre_emb_flat.size(0)

        # Flatten batch and sequence dimensions for post_emb
        B_post, Seq_len_post, D_post = post_emb.shape
        post_emb_flat = post_emb.reshape(-1, D_post)  # [B * Seq_len_post, D]
        curr_num_post = post_emb_flat.size(0)

        # Initialize accumulators on first batch (match device)
        if sum_pre is None:
            device = pre_emb.device
            sum_pre = torch.zeros(D_pre, device=device)
            sum_sq_pre = torch.zeros(D_pre, device=device)
            sum_post = torch.zeros(D_post, device=device)
            sum_sq_post = torch.zeros(D_post, device=device)

        # Update accumulators for pre_emb
        sum_pre += torch.sum(pre_emb_flat, dim=0)
        sum_sq_pre += torch.sum(pre_emb_flat ** 2, dim=0)
        num_samples_pre += curr_num_pre

        # Update accumulators for post_emb
        sum_post += torch.sum(post_emb_flat, dim=0)
        sum_sq_post += torch.sum(post_emb_flat ** 2, dim=0)
        num_samples_post += curr_num_post

    # Compute mean and std for pre_emb
    mean_pre = sum_pre / num_samples_pre
    std_pre = torch.sqrt((sum_sq_pre / num_samples_pre) - (mean_pre ** 2) + 1e-8)

    # Compute mean and std for post_emb
    mean_post = sum_post / num_samples_post
    std_post = torch.sqrt((sum_sq_post / num_samples_post) - (mean_post ** 2) + 1e-8)

    return {
        'mean_pre': mean_pre,
        'std_pre': std_pre,
        'mean_post': mean_post,
        'std_post': std_post
    }

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_folder", type=str,
                        default="/ceph/hpc/data/d2024d05-018-users/wenyan/data/VisU/llava_vector_blocks",
                        help="path to load pre-obtained model embeddigns")
    parser.add_argument("--out_dir", type=str, 
                        default="/ceph/hpc/data/d2024d05-018-users/wenyan/data/VisU/embed_reconstruction_llava_v2")
    parser.add_argument("--embed_model", type=str, default="llava")
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--test_vector_folder", type=str,
                        default="/ceph/hpc/data/d2024d05-018-users/wenyan/data/VisU/llava_vector_blocks/test")
    parser.add_argument("--model_dir", type=str, default="/ceph/hpc/data/d2024d05-018-users/wenyan/data/VisU/embed_reconstruction_llava",
                        help="path to trained checkpoint")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--model_type", type=str, default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--normalize", action="store_true", default=False)
    
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    setup(local_rank, world_size)
    
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    
    # init wandb
    if local_rank == 0:
        run_name = f"{args.embed_model}_reconstruction_{args.model_type}_bs{args.batch_size}_lr{args.lr}_hidden{args.hidden_size}"
        wandb.init(project="embedding-transformer", name=run_name)
    
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    
    if args.eval:
        # Evaluate the model
        pre_embed_store = FlatFileVectorBlockStore(args.test_vector_folder, max_cache_size=args.batch_size*2, prefix="pre")
        post_embed_store = FlatFileVectorBlockStore(args.test_vector_folder, max_cache_size=args.batch_size*2, prefix="post")
        
        paired_vector_dataset = PairedVectorBlockDataset(pre_embed_store, post_embed_store, normalize_vectors=args.normalize)
        
        test_dataloader = DataLoader(
            paired_vector_dataset,
            batch_size=4,
            shuffle=False
        )
        
        # Initialize and load the model
        if args.embed_model == "llava":
            if args.model_type == "mlp":
                model = EmbeddingTransformer(
                    input_dim=4096, 
                    output_dim=1024, 
                    hidden_dim=config.hidden_dim,
                    num_layers=args.num_layers
                ).to(device)
            else:
                model = LargerEmbeddingTransformer(
                    input_dim=4096, 
                    output_dim=1024, 
                    hidden_dim=config.hidden_dim
                ).to(device)
        elif args.embed_model == "idefics2":
            model = SeqEmbeddingTransformer(
                input_dim=4096, 
                input_seq_len=64, 
                output_seq_len=576, 
                output_dim=1152,
                num_layers=16, 
                hidden_dim=config.hidden_dim
            ).to(device)
            print(utils.get_model_params(model))
            
            # init weights
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                        
            model.apply(_init_weights)
        elif args.embed_model == "qwen":
            model = SeqEmbeddingTransformer(
                input_dim=3584, 
                input_seq_len=144, 
                output_seq_len=576, 
                output_dim=1280,
                num_layers=16, 
                hidden_dim=config.hidden_dim
            ).to(device)
            # init weights
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                        
            model.apply(_init_weights)
        else:
            print("model not supported")
            
            
        # Evaluate the model
        stats, sample_results = evaluate_model(model, test_dataloader, device)
        
        # Print evaluation results
        print("\nEvaluation Results:")
        print("-" * 50)
        for metric, value in stats.items():
            print(f"{metric}: {value:.6f}")
        
        # Save detailed results
        results = {
            'statistics': stats,
            'sample_results': sample_results
        }
    
        # Save results in different formats
        output_dir = args.out_dir
        os.makedirs(output_dir, exist_ok=True)
    
        # Save full results (including predictions and targets)
        full_results_file = os.path.join(output_dir, 'full_evaluation_results.pkl')
        with open(full_results_file, 'wb') as f:
            pickle.dump(results, f)
    
        # Save per-sample losses in a more readable format
        per_sample_loss = np.mean(sample_results['sample_losses'], axis=1).tolist()
        per_sample_results = pd.DataFrame({
            'block_id': [x.item() for x in sample_results['block_ids']],
            'loss': per_sample_loss
        })
        per_sample_results.to_csv(os.path.join(output_dir, f'{args.embed_model}_per_sample_losses.csv'), index=False)
        
        print(f"\nFull results saved to {full_results_file}")
        print(f"Per-sample losses saved to {os.path.join(output_dir, f'{args.embed_model}_per_sample_losses.csv')}")
        
        plt.figure(figsize=(10, 6))
        plt.hist(per_sample_loss, bins=50)
        plt.title('Distribution of Reconstruction Losses')
        plt.xlabel('Loss')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f'{args.embed_model}_loss_distribution.png'))
        plt.close()
    
    else:
        # Train the model
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    
        if local_rank == 0:
            config = wandb.config
        else:
            # Create a simple namespace for non-main processes
            from types import SimpleNamespace
            config = SimpleNamespace()

        config.batch_size = args.batch_size  # Reduced batch size
        config.num_epochs = args.epochs
        config.learning_rate = args.lr
        config.hidden_dim = args.hidden_size # default 2048
        config.lr_factor = 0.5
        config.lr_patience = 5
        config.log_interval = 10
        
        config.warmup_steps = 2000
        config.mixed_precision = True
        config.num_layers = args.num_layers
        
        # save config
        if local_rank == 0:
            with open(os.path.join(args.out_dir, "config.pkl"), "wb") as f:
                pickle.dump(config, f)
        
        
        # read preprocessed data
        pre_embed_store = FlatFileVectorBlockStore(args.vector_folder, max_cache_size=args.batch_size*4, prefix="pre")
        post_embed_store = FlatFileVectorBlockStore(args.vector_folder, max_cache_size=args.batch_size*4, prefix="post")
            
        paired_vector_dataset = PairedVectorBlockDataset(pre_embed_store, post_embed_store, normalize_vectors=args.normalize)
        
        total_length = len(paired_vector_dataset)
        train_length = int(0.8 * total_length)
        test_length = total_length - train_length  # ensures lengths sum to total
        print("total number of training vectors: ", train_length)
        print()
        
        
        # Perform the split
        train_dataset, val_dataset = torch.utils.data.random_split(
            paired_vector_dataset, 
            [train_length, test_length]
        )
        
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False
        )
        
        per_gpu_batch = args.batch_size // world_size
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=per_gpu_batch,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=per_gpu_batch,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )


        if args.embed_model == "llava":
            if args.model_type == "mlp":
                model = EmbeddingTransformer(
                    input_dim=4096, 
                    output_dim=1024, 
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers
                ).to(device)
            else:
                model = LargerEmbeddingTransformer(
                    input_dim=4096, 
                    output_dim=1024, 
                    hidden_dim=config.hidden_dim
                ).to(device)
        elif args.embed_model == "idefics2":
            model = SeqEmbeddingTransformer(
                input_dim=4096, 
                input_seq_len=64, 
                output_seq_len=576, 
                output_dim=1152,
                num_layers=args.num_layers, 
                hidden_dim=config.hidden_dim
            ).to(device)
            
            # init weights
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                        
            model.apply(_init_weights)
            print(model)
        elif args.embed_model == "qwen":
            model = SeqEmbeddingTransformer(
                input_dim=3584, 
                input_seq_len=144, 
                output_seq_len=576, 
                output_dim=1280,
                num_layers=args.num_layers, 
                hidden_dim=config.hidden_dim
            ).to(device)
            # init weights
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                        
            model.apply(_init_weights)
        else:
            print("model not supported")
            
            
        # Wrap model with DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if local_rank == 0:
            total_params, trainable_params = get_model_params(model)
            print(f"\nTotal Parameters: {total_params:,}")
            print(f"Trainable Parameters: {trainable_params:,}")

        best_model_state = train_model(model, train_dataloader, val_dataloader, config, args, 
                                        local_rank, world_size,device=device)
        if local_rank == 0:
            torch.save(best_model_state, os.path.join(args.out_dir, 'best_embedding_transformer.pth'))
        # Clean up
        dist.destroy_process_group()
        torch.cuda.empty_cache()
if __name__ == "__main__":
    main()