import torch
import random
import numpy as np
import pandas as pd

from models.modeling_dsm import DSM


MODEL_PATH = 'Synthyra/DSM_ppi_full'
TEMPERATURE = 1.0
REMASKING = 'random'
SLOW = False
PREVIEW = False
STEP_DIVISOR = 10
BATCH_SIZE = 1

SEQA = 'LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPS'
SEQB = 'QVQLQQSGPGLVQPSQSLSITCTVSGFSLTNYGVHWVRQSPGKGLEWLGVIWSGGNTDYNTPFTSRLSISRDTSKSQVFFKMNSLQTDDTAIYYCARALTYYDYEFAYWGQGTLVTVSAGGGGSGGGGSGGGGSDILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPKLLIRYASESISGIPSRFSGSGSGTDFTLSINSVDPEDIADYYCQQNNNWPTTFGAGTKLELK'


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


def redesign_pair(model, tokenizer, seq_a, seq_b):
    """Process PPI examples and calculate reconstruction accuracy"""
    mask_token = '<mask>'
    eos_token = tokenizer.eos_token
    # Mask 50% of SeqB
    masked_seq_a = seq_a
    masked_seq_b = mask_sequence(seq_b, mask_ratio=0.9, mask_token=mask_token)[0]
    template = masked_seq_a + eos_token + masked_seq_b
    template_tokens = tokenizer.encode(template, add_special_tokens=True, return_tensors='pt').to(model.device)
    num_mask_tokens = (template_tokens == tokenizer.mask_token_id).sum().item()

    #print(tokenizer.decode(template_tokens[0]).replace(' ', ''))
    # Generate reconstruction
    output_tokens, trajectory = model.mask_diffusion_generate(
        template_tokens=template_tokens,
        block_wise=False,
        batch_size=BATCH_SIZE,
        steps=max(1, num_mask_tokens // STEP_DIVISOR),
        temperature=TEMPERATURE,
        remasking=REMASKING,
        preview=PREVIEW,
        slow=SLOW,
        start_with_methionine=False,
        return_trajectory=True
    )
    for seq in trajectory:
        print(seq)
        print('-' * 100)

    # Decode the generated sequence
    output_tokens = output_tokens[0]
    reconstructed_full = model._decode_seq(output_tokens)
    
    # Extract the reconstructed SeqB part (after SeqA)
    new_a = reconstructed_full[:len(seq_a)]
    new_b = reconstructed_full[len(seq_a):]

    return new_a, new_b, trajectory


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
    new_a, new_b, trajectory = redesign_pair(model, tokenizer, SEQA, SEQB)
    print(new_a)
    print(new_b)


    seqs_a, seqs_b = [], []
    for seq in trajectory:
        seq = seq.replace('-', 'A')
        seq_a, seq_b = seq[:len(SEQA)], seq[len(SEQA):]
        seqs_a.append(seq_a)
        seqs_b.append(seq_b)
    
    df = pd.DataFrame({
        'SeqA': seqs_a,
        'SeqB': seqs_b
    })
    df.to_csv('trajectory.csv', index=False)