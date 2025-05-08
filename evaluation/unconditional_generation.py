import torch
import argparse
import os
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from huggingface_hub import login, hf_hub_download

from models.modeling_esm_diff import ESM_Diff
from evaluation.compare_distributions import CorpusComparator, AA20


MODEL_PATH = 'GleghornLab/ESM_diff_650'

TEMPERATURE = 1.0
REMASKING = 'random'
SLOW = False
PREVIEW = False
STEP_DIVISOR = 100


def get_eval_data():
    local_file = hf_hub_download(
        repo_id="Synthyra/omg_prot50",
        filename=f"data/valid-00000-of-00001.parquet",
        repo_type="dataset"
    )
    data = Dataset.from_parquet(local_file).shuffle(seed=888).select(range(100))
    data = data.filter(lambda x: len(x['sequence']) > 20 and len(x['sequence']) < 2048)
    print(data)
    valid_seqs = data['sequence']
    return valid_seqs


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--output_path', type=str, default='unconditional_generation_seqs.csv', help='Output path')
    return parser.parse_args()


if __name__ == '__main__':
    # py -m evaluation.unconditional_generation
    args = arg_parser()
    if args.token is not None:
        login(args.token)
    
    args.output_path = os.path.join('evaluation', 'comparisons', args.output_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESM_Diff.from_pretrained(MODEL_PATH).to(device).eval()
    tokenizer = model.tokenizer

    natural_seqs = get_eval_data()

    generated_seqs = []

    for seq in tqdm(natural_seqs):
        output_tokens = model.mask_diffusion_generate(
            length=len(seq),
            block_wise=False,
            batch_size=args.batch_size,
            steps=len(seq) // STEP_DIVISOR,
            temperature=TEMPERATURE,
            remasking=REMASKING,
            preview=PREVIEW,
            slow=SLOW,
            start_with_methionine=False
        )
        for output_token in output_tokens:
            gen_seq = model._decode_seq(output_token)
            assert len(gen_seq) == len(seq), f'Differing lengths: {len(gen_seq)} != {len(seq)}'
            assert gen_seq.count('-') == 0, f'Masks present: {gen_seq.count("-")} != 0'

            #if len(gen_seq) != len(seq):
            #    print(f'WARNING: Differing lengths: {len(gen_seq)} != {len(seq)}')
            #if gen_seq.count('-') != 0:
            #    print(f'WARNING: Gaps present: {gen_seq.count("-")} != 0')
            generated_seqs.append(gen_seq)

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
