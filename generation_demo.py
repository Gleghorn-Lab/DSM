import torch
import random
from tqdm import tqdm

from models.modeling_dsm import DSM


MODEL_PATH = 'GleghornLab/DSM_650'
TEMPERATURE = 1.0
REMASKING = 'random'
SLOW = False
PREVIEW = True
STEP_DIVISOR = 10
BATCH_SIZE = 1
NUM_SEQS = 10000


if __name__ == '__main__':
    # py -m generation_demo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DSM.from_pretrained(MODEL_PATH).to(device).eval()
    tokenizer = model.tokenizer

    for _ in range(NUM_SEQS):
        seq_length = random.randint(20, 100)
        output_tokens = model.mask_diffusion_generate(
            length=seq_length,
            block_wise=False,
            batch_size=BATCH_SIZE,
            steps=seq_length // STEP_DIVISOR,
            temperature=TEMPERATURE,
            remasking=REMASKING,
            preview=PREVIEW,
            slow=SLOW,
            start_with_methionine=False
        )
        output_tokens = output_tokens[0]
        generated_seq = model._decode_seq(output_tokens)

        for token in output_tokens:
            if token == model.cls_token_id:
                continue
            if token == model.eos_token_id:
                continue
            if token.item() not in model.canonical_amino_acid_ids:
                raise ValueError(f'{token} found')



