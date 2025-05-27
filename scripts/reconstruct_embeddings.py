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
from reconstruct_embeddings_parallel import EmbeddingTransformer, compute_dataset_stats, standardize_embeddings


class PairedVectorBlock(TypedDict):
    embeddings: tuple[torch.Tensor, torch.Tensor]
    block_id: int
    

class PairedVectorBlockDataset(Dataset):
    def __init__(self, block_store1: VectorBlockStore, block_store2: VectorBlockStore) -> None:
        """
        A dataset that returns `num_pairs` pairs of embeddings from the two block stores.
        """
        assert len(block_store1.block_ids) == len(block_store2.block_ids), "must be same number of blocks"
        assert set(block_store1.block_ids) == set(block_store2.block_ids), "must be same blocks"
        self.block_store1 = block_store1
        self.block_store2 = block_store2

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
                nhead=16,
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



def train_model(model, train_loader, val_loader, config, args, device='cuda'):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Main scheduler
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs * len(train_loader) - config.warmup_steps,
        eta_min=config.learning_rate * 0.01
    )
    
    # Warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / config.warmup_steps)
    )
    
    # Combined scheduler
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[config.warmup_steps]
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler(enabled=config.mixed_precision)
    
    best_val_loss = float('inf')
    best_model = None
    
    # Early stopping parameters
    patience = 10  # Number of epochs to wait for improvement
    patience_counter = 0
    min_delta = 1e-3  # Minimum change in validation loss to qualify as an improvement
    
    # Gradient accumulation steps
    accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps
    
    for epoch in range(config.num_epochs):
        
        # Training phase
        model.train()
        train_loss = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Training]", ncols=100)
        
        for batch_idx, batch in enumerate(train_loader_tqdm):
            post_emb = batch["embeddings"][1].to(device)
            pre_emb = batch["embeddings"][0].to(device)
            
            with autocast():
                output = model(post_emb)
                loss = criterion(output, pre_emb)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            train_loss += loss.item() * accumulation_steps
            train_loader_tqdm.set_postfix({"Batch Loss": loss.item() * accumulation_steps})
            
            if batch_idx % config.log_interval == 0:
                wandb.log({
                    "batch_loss": loss.item() * accumulation_steps,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                    "batch": batch_idx
                })
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Validation]", ncols=100)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader_tqdm):
                post_emb = batch["embeddings"][1].to(device, non_blocking=True)
                pre_emb = batch["embeddings"][0].to(device, non_blocking=True)
                
                with autocast():
                    output = model(post_emb)
                    loss = criterion(output, pre_emb)
                
                val_loss += loss.item()
                val_loader_tqdm.set_postfix({"Val Loss": loss.item()})
                
                del output, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            val_loss /= len(val_loader)
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            print(f"Validation loss decreased ({best_val_loss:.6f} --> {val_loss:.6f})")
            best_val_loss = val_loss
            best_model = model.state_dict()
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
            wandb.save("best_model.pt")
            patience_counter = 0  # Reset counter
            print("model saved to ", os.path.join(args.out_dir, "best_model.pt"))
        else:
            patience_counter += 1
            
        # Log early stopping info
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
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    return best_model


def evaluate_model(model, test_loader, pred_vector_store, args, device='cuda'):
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    total_loss = 0
    
    # Store detailed results for each sample
    sample_results = {
        'block_ids': [],
        'sample_losses': [],
        'feature_losses': [],
        'predictions': [],
        'targets': []
    }
    
    
    if args.normalize:
        print("calculate dataset mean and std")
        stats_dict = compute_dataset_stats(test_loader)
        with open(os.path.join(args.out_dir, "dataset_stats.pkl"), "wb") as f:
            pickle.dump(stats_dict, f)
        
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if args.normalize:
                post_emb = standardize_embeddings(batch["embeddings"][1], stats_dict["mean_post"], stats_dict["std_post"]).to(device)
                pre_emb = standardize_embeddings(batch["embeddings"][0], stats_dict["mean_pre"], stats_dict["std_pre"]).to(device)
            else:
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
                for idx in range(output.size(0)): # save the predicted vectors
                    pred_vector_store.store(block_ids[idx], output[idx].cpu())
                
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

def get_state_dict(args):
    if os.path.exists(os.path.join(args.model_dir, "best_embedding_transformer.pth")):
        state_dict = torch.load(os.path.join(args.model_dir, "best_embedding_transformer.pth"))
    else:
        state_dict = torch.load(os.path.join(args.model_dir, "best_model.pt"))
    
    #remove module from key
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return state_dict

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
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true", default=False)
    args = parser.parse_args()
    
    # init wandb
    run_name = f"{args.embed_model}_reconstruction_{args.model_type}_bs{args.batch_size}_lr{args.lr}_hidden{args.hidden_size}"
    wandb.init(project="embedding-transformer", name=run_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    
    if args.eval:
        # Evaluate the model
        pre_embed_store = FlatFileVectorBlockStore(args.test_vector_folder, max_cache_size=100, prefix="pre")
        post_embed_store = FlatFileVectorBlockStore(args.test_vector_folder, max_cache_size=100, prefix="post")
        
        paired_vector_dataset = PairedVectorBlockDataset(pre_embed_store, post_embed_store)
        
        pred_vector_dir = os.path.join(args.out_dir, "pred_vectors")
        os.makedirs(pred_vector_dir, exist_ok=True)
        pred_vector_store = FlatFileVectorBlockStore(pred_vector_dir, max_cache_size=100, prefix="pred")
        
        test_dataloader = DataLoader(
            paired_vector_dataset,
            batch_size=32,
            shuffle=False
        )
        
        # Initialize and load the model
        if args.embed_model == "llava":
            if args.model_type == "mlp":
                model = EmbeddingTransformer(
                    input_dim=4096, 
                    output_dim=1024, 
                    hidden_dim=args.hidden_size,
                    num_layers=args.num_layers
                )
            else:
                model = LargerEmbeddingTransformer(
                    input_dim=4096, 
                    output_dim=1024, 
                    hidden_dim=args.hidden_size
                )
            model.load_state_dict(get_state_dict(args))
            print("model loaded")
            model = model.to(device)
            print(utils.get_model_params(model))
        elif args.embed_model == "idefics2":
            model = SeqEmbeddingTransformer(
                input_dim=4096, 
                input_seq_len=64, 
                output_seq_len=576, 
                output_dim=1152,
                num_layers=args.num_layers,
                hidden_dim=args.hidden_size
            ).to(device)
            model.load_state_dict(get_state_dict(args))
        elif args.embed_model == "qwen":
            model = SeqEmbeddingTransformer(
                input_dim=3584,
                input_seq_len=144,
                output_seq_len=576,
                output_dim=1280,
                num_layers=args.num_layers
            ).to(device)
        else:
            print("model not supported")
            
            
        # Evaluate the model
        stats, sample_results = evaluate_model(model, test_dataloader, pred_vector_store, args, device)
        
        # Print evaluation results
        print("\nEvaluation Results:")
        print("-" * 50)
        for metric, value in stats.items():
            print(f"{metric}: {value:.6f}")
        
        # Save detailed results
        del sample_results["predictions"]
        del sample_results["targets"]
        
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
        config = wandb.config
        config.batch_size = args.batch_size  # Reduced batch size
        config.num_epochs = args.epochs
        config.learning_rate = args.lr
        config.hidden_dim = args.hidden_size # default 2048
        config.lr_factor = 0.5
        config.lr_patience = 5
        config.log_interval = 10
        
        config.warmup_steps = 500
        config.mixed_precision = True
        
        
        # read preprocessed data
        pre_embed_store = FlatFileVectorBlockStore(args.vector_folder, max_cache_size=args.batch_size*4, prefix="pre")
        post_embed_store = FlatFileVectorBlockStore(args.vector_folder, max_cache_size=args.batch_size*4, prefix="post")
            
        paired_vector_dataset = PairedVectorBlockDataset(pre_embed_store, post_embed_store)
        
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
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=4
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )

        if args.embed_model == "llava":
            if args.model_type == "default":
                model = EmbeddingTransformer(
                    input_dim=4096, 
                    output_dim=1024, 
                    hidden_dim=config.hidden_dim
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
            
        total_params, trainable_params = get_model_params(model)
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

        best_model_state = train_model(model, train_dataloader, val_dataloader, config, args, device=device)
        torch.save(best_model_state, os.path.join(args.out_dir, 'best_embedding_transformer.pth'))

if __name__ == "__main__":
    main()