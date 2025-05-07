import torch
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from huggingface_hub import login, hf_hub_download

from models.modeling_esm_diff import ESM_Diff
from evaluation.compare_distributions import compare_corpora_kmers


MODEL_PATH = 'GleghornLab/ESM_diff_650'

TEMPERATURE = 1.0
REMASKING = 'random'
SLOW = False
PREVIEW = False
STEP_DIVISOR = 5


def get_eval_data():
    local_file = hf_hub_download(
        repo_id="Synthyra/omg_prot50",
        filename=f"data/valid-00000-of-00001.parquet",
        repo_type="dataset"
    )
    data = Dataset.from_parquet(local_file).shuffle(seed=888)
    data = data.filter(lambda x: len(x['sequence']) > 20 and len(x['sequence']) < 2048)
    print(data)
    valid_seqs = data['sequence']
    return valid_seqs


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    return parser.parse_args()


if __name__ == '__main__':
    # py -m evaluation.unconditional_generation
    args = arg_parser()
    if args.token is not None:
        login(args.token)
    
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
        for seq in output_tokens:
            generated_seqs.append(model._decode_seq(seq))

    stats = compare_corpora_kmers(natural_seqs, generated_seqs)

    result = []

    for k, res in stats.items():
        chi_p = f'{res["p"]:.3g}'
        jsd = f'{res["js"]:.4f}'
        result.append((k, chi_p, jsd))

    ### Write natural and generated sequences to csv
    df = pd.DataFrame()
    df['natural'] = natural_seqs
    df['generated'] = generated_seqs
    df.to_csv('unconditional_generation_seqs.csv', index=False)
