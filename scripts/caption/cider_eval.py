import json
import numpy as np
from pycocoevalcap.cider.cider import Cider
from tqdm import tqdm
import argparse
import os
from collections import defaultdict

def load_results(results_file):
    """Load results from a JSONL file."""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results

def calculate_cider_scores(results):
    """Calculate CIDEr score for each sample and overall average."""
    # Prepare data in the format expected by pycocoevalcap
    gts = {}  # Ground truth captions
    res = {}  # Generated captions
    
    for i, item in enumerate(results):
        image_id = item['image_id']  # Use image_id as the identifier
        
        # Skip samples with empty generated captions or references
        if not item['generated_caption'] or not item['reference_captions']:
            continue
            
        # The issue is with the dictionary format - CIDEr expects raw strings
        # Convert references to strings
        gts[image_id] = item['reference_captions']
        # Convert hypothesis to string
        res[image_id] = [item['generated_caption'].strip("\n")]
    
    # Initialize CIDEr scorer
    cider_scorer = Cider()
    
    # Calculate overall CIDEr score
    overall_score, individual_scores = cider_scorer.compute_score(gts, res)
    
    # Create a mapping from image_id to its CIDEr score
    score_map = {img_id: score for img_id, score in zip(gts.keys(), individual_scores)}
    
    return overall_score, score_map

def main():
    parser = argparse.ArgumentParser(description='Calculate CIDEr scores for caption generation results')
    parser.add_argument('--results_file', type=str, required=True, help='Path to the JSONL results file')
    parser.add_argument('--output_file', type=str, default=None, help='Path to save per-sample CIDEr scores')
    parser.add_argument('--detailed_output', action='store_true', help='Include detailed output with reference captions')
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_file}...")
    results = load_results(args.results_file)
    print(f"Loaded {len(results)} samples")
    
    # Calculate CIDEr scores
    print("Calculating CIDEr scores...")
    overall_cider, individual_cider_scores = calculate_cider_scores(results)
    print(f"Overall CIDEr score: {overall_cider:.4f}")
    
    # Add CIDEr scores to results
    for item in results:
        item['cider_score'] = individual_cider_scores.get(item['image_id'], None)
    
    # Calculate statistics on scores
    valid_scores = [item['cider_score'] for item in results if item['cider_score'] is not None]
    if valid_scores:
        print(f"Min CIDEr: {min(valid_scores):.4f}")
        print(f"Max CIDEr: {max(valid_scores):.4f}")
        print(f"Mean CIDEr: {np.mean(valid_scores):.4f}")
        print(f"Median CIDEr: {np.median(valid_scores):.4f}")
    
    # Save per-sample CIDEr scores if output file is specified
    output_file = args.results_file.replace(".jsonl", "_cider.jsonl") if args.output_file is None else args.output_file
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_file, 'w') as f:
            for item in results:
                if args.detailed_output:
                    # Include all details
                    output_item = item
                else:
                    # Only include image_id, generated caption, and CIDEr score
                    output_item = {
                        'image_id': item['image_id'],
                        'generated_caption': item['generated_caption'],
                        'cider_score': item['cider_score']
                    }
                f.write(json.dumps(output_item) + '\n')
        print(f"Per-sample CIDEr scores saved to {output_file}")
    
    # Analyze score distribution
    if valid_scores:
        score_bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, float('inf')]
        bin_counts = defaultdict(int)
        
        for score in valid_scores:
            for i in range(len(score_bins) - 1):
                if score_bins[i] <= score < score_bins[i+1]:
                    bin_counts[f"{score_bins[i]}-{score_bins[i+1]}"] += 1
                    break
        
        print("\nCIDEr Score Distribution:")
        for bin_range, count in sorted(bin_counts.items()):
            percentage = (count / len(valid_scores)) * 100
            print(f"{bin_range}: {count} samples ({percentage:.2f}%)")

if __name__ == "__main__":
    main()