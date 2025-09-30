import torch
import random
from tqdm.auto import tqdm
from models.modeling_dsm import DSM


# Load a pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DSM.from_pretrained("Synthyra/DSM_ppi_full").to(device).eval()
tokenizer = model.tokenizer
mask_token = tokenizer.mask_token

batch_size = 16  # adjust as needed

for i in tqdm(range(1000)):
    # Random lengths per example between 20 and 1000
    seq_a_lengths = [random.randint(20, 1000) for _ in range(batch_size)]
    seq_b_lengths = [random.randint(20, 1000) for _ in range(batch_size)]

    # Build batched masked templates
    combined_inputs = [
        (mask_token * la) + '<eos>' + (mask_token * lb)
        for la, lb in zip(seq_a_lengths, seq_b_lengths)
    ]

    tokenized = tokenizer(
        combined_inputs,
        add_special_tokens=True,
        padding=True,
        return_tensors='pt'
    )
    input_tokens = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    output = model.mask_diffusion_generate(
        tokenizer=tokenizer,
        input_tokens=input_tokens,
        attention_mask=attention_mask,
        step_divisor=10,          # lower is slower but better
        temperature=1.0,          # sampling temperature
        remasking="random",       # strategy for remasking tokens not kept
        preview=False,            # set this to True to watch the mask tokens get filled in real time
        slow=False,               # adds a small delay to the real time filling (because it is usually very fast and watching carefully is hard!)
        return_trajectory=False   # set this to True to return the trajectory of the generation (what you watch in the preview)
    ) # Note: output will be a tuple if return_trajectory is True

    seqa, seqb = model.decode_dual_input(output, attention_mask=attention_mask, seperator='<eos>')
    for a, b in zip(seqa, seqb):
        assert '<' not in a and '<' not in b and '>' not in a and '>' not in b, f"Found special characters in \n{a}\n{b}"