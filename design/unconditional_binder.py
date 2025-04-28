import torch
import argparse
from huggingface_hub import login
from models.modeling_esm_diff import ESM_Diff
from models.alignment_helpers import analyze_two_seqs


SYNTHYRA_API_KEY = ''

MODEL_PATH = 'GleghornLab/eval_diff_150'
TARGET = 'LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPS'
TARGET_AMINOS = ['S11', 'N12', 'K13', 'T15', 'Q16', 'L17', 'G18', 'S356', 'S440', 'G441']
TARGET_IDX = [11, 12, 13, 15, 16, 17, 18, 356, 440, 441]
TEMPLATE = 'QVQLQQSGPGLVQPSQSLSITCTVSGFSLTNYGVHWVRQSPGKGLEWLGVIWSGGNTDYNTPFTSRLSISRDTSKSQVFFKMNSLQTDDTAIYYCARALTYYDYEFAYWGQGTLVTVSAGGGGSGGGGSGGGGSDILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPKLLIRYASESISGIPSRFSGSGSGTDFTLSINSVDPEDIADYYCQQNNNWPTTFGAGTKLELK'


parser = argparse.ArgumentParser()
parser.add_argument('--token', type=str, default=None)


if __name__ == '__main__':
    # py -m design.unconditional_binder
    
    args = parser.parse_args()
    if args.token is None:
        login(args.token)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESM_Diff.from_pretrained(MODEL_PATH).to(device)
    tokenizer = model.tokenizer


    steps = 4
    block_length = steps // 4
    temperature = 1.0
    remasking = 'random'
    slow = False
    preview = True
    mask_percentage = 0.25

    template_tokens = tokenizer.encode(TEMPLATE, add_special_tokens=True, return_tensors='pt').to(device)
    # randomly mask 25% of the template tokens, do not mask CLS or EOS tokens
    mask_index = torch.rand_like(template_tokens.float()) < mask_percentage
    mask_index[:, 0], mask_index[:, -1] = False, False
    template_tokens[mask_index] = tokenizer.mask_token_id


    output_tokens = model.mask_diffusion_generate(
        template_tokens=template_tokens,
        block_wise=False,
        steps=steps,
        temperature=temperature,
        remasking=remasking,
        preview=preview,
        slow=slow,
        start_with_methionine=False
    )

    prediction = model._decode_seq(output_tokens[0])
    print(prediction)

    analyze_two_seqs(TARGET, prediction)
