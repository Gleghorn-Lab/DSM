import torch
import argparse
import os
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login

from models.modeling_dsm import DSM
from evaluation.compare_distributions import CorpusComparator, AA20
from .utils import get_eval_data


MODEL_PATH = 'GleghornLab/DSM_650'
TEMPERATURE = 1.0
REMASKING = 'random'
SLOW = False
PREVIEW = True


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--output_path', type=str, default='sequences_by_length.csv', help='Output path')
    parser.add_argument('--step_divisor', type=int, default=1, help='Step divisor')
    parser.add_argument('--min_length', type=int, default=100, help='Minimum sequence length')
    parser.add_argument('--max_length', type=int, default=2000, help='Maximum sequence length')
    parser.add_argument('--length_step', type=int, default=100, help='Length step size')
    return parser.parse_args()


if __name__ == '__main__':
    # py -m evaluation.unconditional_by_length
    args = arg_parser()
    if args.token is not None:
        login(args.token)
    
    args.output_path = os.path.join('evaluation', 'comparisons', args.output_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DSM.from_pretrained(MODEL_PATH).to(device).eval()
    tokenizer = model.tokenizer

    sequence_lengths = range(args.min_length, args.max_length + 1, args.length_step)
    generated_sequences = {}

    for length in tqdm(sequence_lengths, desc="Generating sequences"):
        output_tokens = model.mask_diffusion_generate(
            length=length,
            block_wise=False,
            batch_size=args.batch_size,
            steps=length // args.step_divisor,
            temperature=TEMPERATURE,
            remasking=REMASKING,
            preview=PREVIEW,
            slow=SLOW,
            start_with_methionine=False
        )
        gen_seq = model._decode_seq(output_tokens[0])
        assert len(gen_seq) == length, f'Differing lengths: {len(gen_seq)} != {length}'
        assert gen_seq.count('-') == 0, f'Masks present: {gen_seq.count("-")} != 0'
        generated_sequences[length] = gen_seq
        print(f"Generated sequence of length {length}")
        print(gen_seq)

    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        'length': list(generated_sequences.keys()),
        'sequence': list(generated_sequences.values())
    })
    
    df.to_csv(args.output_path, index=False)
    print(f"Saved sequences to {args.output_path}")
