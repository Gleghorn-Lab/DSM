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
PREVIEW = False


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--output_path', type=str, default='unconditional_generation_seqs.csv', help='Output path')
    parser.add_argument('--step_divisor', type=int, default=5, help='Step divisor')
    return parser.parse_args()


if __name__ == '__main__':
    # py -m evaluation.unconditional_generation
    args = arg_parser()
    if args.token is not None:
        login(args.token)
    
    args.output_path = os.path.join('evaluation', 'comparisons', args.output_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DSM.from_pretrained(MODEL_PATH).to(device).eval()
    tokenizer = model.tokenizer
    mask_token = tokenizer.mask_token

    natural_seqs = get_eval_data()

    generated_seqs = []

    for seq in tqdm(natural_seqs):
        template = ''.join([mask_token] * len(seq))
        template_tokens = tokenizer.encode(template, add_special_tokens=True, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(template_tokens)

        output_tokens = model.mask_diffusion_generate(
            tokenizer=tokenizer,
            input_tokens=template_tokens,
            step_divisor=args.step_divisor,
            temperature=TEMPERATURE,
            remasking=REMASKING,
            preview=PREVIEW,
            slow=SLOW,
            start_with_methionine=False
        )
        generated_seqs.extend(model.decode_output(output_tokens, attention_mask))

    comparator = CorpusComparator(vocabulary=AA20)
    stats = comparator.compare_corpora_kmers(natural_seqs, generated_seqs)

    result = []

    for k, res in stats.items():
        chi_p = f'{res["p"]:.3g}'
        jsd = f'{res["js"]:.4f}'
        result.append((k, chi_p, jsd))

    ### Write natural and generated sequences to csv
    df = pd.DataFrame()
    df['natural'] = natural_seqs
    df['generated'] = generated_seqs
    df.to_csv(args.output_path, index=False)
