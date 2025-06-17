import torch
import random

from evaluation.utils import get_eval_data, load_binder_model


MODEL_PATH = 'GleghornLab/DSM_650_ppi'
TEMPERATURE = 1.0
REMASKING = 'random'
SLOW = False
PREVIEW = True
STEP_DIVISOR = 10
BATCH_SIZE = 1
NUM_SEQS = 10


if __name__ == '__main__':
    # py -m generation_demo_ppi
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_binder_model(MODEL_PATH).to(device).eval()
    tokenizer = model.tokenizer

    natural_seqs = get_eval_data(NUM_SEQS)

    mask_token = '<mask>'
    eos_token = tokenizer.eos_token_id

    for seq in natural_seqs:
        seq = seq[:100]
        binder_length = random.randint(20, 100)
        template = ''.join([mask_token] * binder_length)
        target_tokens = tokenizer.encode(seq, add_special_tokens=True, return_tensors='pt').to(device)
        template_tokens = tokenizer.encode(template, add_special_tokens=False, return_tensors='pt').to(device)
        end_eos = torch.tensor([eos_token], device=device).unsqueeze(0)

        # cls, target, eos, template, eos
        template_tokens = torch.cat([target_tokens, template_tokens, end_eos], dim=1)
        output_tokens = model.mask_diffusion_generate(
            tokenizer=tokenizer,
            input_tokens=template_tokens,
            step_divisor=STEP_DIVISOR,
            temperature=TEMPERATURE,
            remasking=REMASKING,
            preview=PREVIEW,
            slow=SLOW,
        )
        a, b = model.decode_dual_input(output_tokens, template_tokens, '<eos>')
        print(a[0])
        print(b[0])
        print('-' * 100)
