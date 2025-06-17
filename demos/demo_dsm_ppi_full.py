import torch
import random
import numpy as np
from tqdm.auto import tqdm

from models.modeling_dsm import DSM
from evaluation.utils import get_ppi_examples


MODEL_PATH = 'Synthyra/DSM_ppi_full'
TEMPERATURE = 1.0
REMASKING = 'random'
SLOW = False
PREVIEW = False
STEP_DIVISOR = 10
BATCH_SIZE = 1
NUM_SEQS = 1000
MAX_COMBINED_LENGTH = 256


def mask_sequence(seq, mask_ratio=0.5, mask_token='<mask>'):
    """Mask a fraction of amino acids in a sequence"""
    seq_list = list(seq)
    num_to_mask = int(len(seq_list) * mask_ratio)
    positions_to_mask = random.sample(range(len(seq_list)), num_to_mask)
    
    original_tokens = []
    for i, pos in enumerate(positions_to_mask):
        original_tokens.append((pos, seq_list[pos]))
        seq_list[pos] = mask_token
    
    return ''.join(seq_list), original_tokens


def calculate_accuracy(original_tokens, reconstructed_seq):
    """Calculate reconstruction accuracy for masked positions"""
    correct = 0
    total = len(original_tokens)
    
    for pos, original_aa in original_tokens:
        if pos < len(reconstructed_seq) and reconstructed_seq[pos] == original_aa:
            correct += 1
    
    return correct / total


def process_ppi_examples(model, tokenizer, seqs_a, seqs_b, example_type):
    """Process PPI examples and calculate reconstruction accuracy"""
    accuracies = []
    mask_token = '<mask>'
    eos_token = tokenizer.eos_token
    
    print(f"\nProcessing {example_type} examples...")
    
    pbar = tqdm(zip(seqs_a, seqs_b), total=len(seqs_a))
    for seq_a, seq_b in pbar:
        # Mask 50% of SeqB
        masked_seq_b, original_tokens = mask_sequence(seq_b, mask_ratio=0.5, mask_token=mask_token)
        
        # Create template: SeqA + masked SeqB
        template = seq_a + eos_token + masked_seq_b
        template_tokens = tokenizer.encode(template, add_special_tokens=True, return_tensors='pt').to(model.device)

        # Generate reconstruction
        output_tokens = model.mask_diffusion_generate(
            tokenizer=tokenizer,
            input_tokens=template_tokens,
            step_divisor=STEP_DIVISOR,
            temperature=TEMPERATURE,
            remasking=REMASKING,
            preview=PREVIEW,
            slow=SLOW,
        )
        
        recon_a, recon_b = model.decode_dual_input(output_tokens, template_tokens, '<eos>')
        
        # Calculate accuracy for masked positions
        accuracy = calculate_accuracy(original_tokens, recon_b[0])
        accuracies.append(accuracy)
        
        pbar.set_description(f"Accuracy: {accuracy:.3f}")
        pbar.update(1)
    
    return accuracies


if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = DSM.from_pretrained(MODEL_PATH).to(device).eval()
    tokenizer = model.tokenizer
    
    # Get PPI examples
    print("Loading PPI examples...")
    positives, negatives = get_ppi_examples(NUM_SEQS, MAX_COMBINED_LENGTH)
    
    # Process positive examples
    positive_accuracies = process_ppi_examples(
        model, tokenizer, 
        positives['SeqA'], positives['SeqB'], 
        "positive"
    )
    
    # Process negative examples  
    negative_accuracies = process_ppi_examples(
        model, tokenizer,
        negatives['SeqA'], negatives['SeqB'],
        "negative"
    )
    
    # Calculate and compare results
    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)
    
    if positive_accuracies:
        pos_mean = np.mean(positive_accuracies)
        pos_std = np.std(positive_accuracies)
        print(f"Positive examples:")
        print(f"  Mean accuracy: {pos_mean:.3f} ± {pos_std:.3f}")
        print(f"  Processed:     {len(positive_accuracies)} examples")
    
    if negative_accuracies:
        neg_mean = np.mean(negative_accuracies)
        neg_std = np.std(negative_accuracies)
        print(f"\nNegative examples:")
        print(f"  Mean accuracy: {neg_mean:.3f} ± {neg_std:.3f}")
        print(f"  Processed:     {len(negative_accuracies)} examples")
    
    if positive_accuracies and negative_accuracies:
        difference = pos_mean - neg_mean
        print(f"\nDifference (Positive - Negative): {difference:.3f}")
        
        # Statistical significance test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(positive_accuracies, negative_accuracies)
        print(f"T-test: t={t_stat:.3f}, p={p_value:.3f}")
        
        if p_value < 0.05:
            print("Difference is statistically significant (p < 0.05)")
        else:
            print("Difference is not statistically significant (p >= 0.05)")
