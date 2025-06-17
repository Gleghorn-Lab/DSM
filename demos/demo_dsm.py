import torch
import random

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
    mask_token = tokenizer.mask_token

    for _ in range(NUM_SEQS):
        seq_length = random.randint(20, 100)
        template = ''.join([mask_token] * seq_length)
        template_tokens = tokenizer.encode(template, add_special_tokens=True, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(template_tokens)

        output_tokens = model.mask_diffusion_generate(
            tokenizer=tokenizer,
            input_tokens=template_tokens,
            step_divisor=STEP_DIVISOR,
            temperature=TEMPERATURE,
            remasking=REMASKING,
            preview=PREVIEW,
            slow=SLOW,
        )
        generated_seq = model.decode_output(output_tokens, attention_mask)[0]
        print(generated_seq)
        print('-' * 100)
