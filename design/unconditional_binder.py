import torch
import random
import argparse
from tqdm import tqdm
from huggingface_hub import login
from models.modeling_esm_diff import ESM_Diff
from .affinity_pred import predict_against_target


SYNTHYRA_API_KEY = '7147b8da62cc094c11d688dbac739e4689cdc7952d5196a488e5d95a6c2f2da1'

MODEL_PATH = 'GleghornLab/esm_diff_150'
TARGET = 'LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPS'
TARGET_AMINOS = ['S11', 'N12', 'K13', 'T15', 'Q16', 'L17', 'G18', 'S356', 'S440', 'G441']
TARGET_IDX = [11, 12, 13, 15, 16, 17, 18, 356, 440, 441]
TEMPLATE = 'QVQLQQSGPGLVQPSQSLSITCTVSGFSLTNYGVHWVRQSPGKGLEWLGVIWSGGNTDYNTPFTSRLSISRDTSKSQVFFKMNSLQTDDTAIYYCARALTYYDYEFAYWGQGTLVTVSAGGGGSGGGGSGGGGSDILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPKLLIRYASESISGIPSRFSGSGSGTDFTLSINSVDPEDIADYYCQQNNNWPTTFGAGTKLELK'
TRUE_PKD = 8.918238


parser = argparse.ArgumentParser()
parser.add_argument('--token', type=str, default=None)
parser.add_argument('--num_samples', type=int, default=1000)


if __name__ == '__main__':
    # py -m design.unconditional_binder
    args = parser.parse_args()
    if args.token is not None:
        login(args.token)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESM_Diff.from_pretrained(MODEL_PATH).to(device)
    tokenizer = model.tokenizer

    temperature = 1.0
    remasking = 'random'
    slow = False
    preview = False

    designs = []

    for sample in tqdm(range(args.num_samples)):
        mask_percentage = random.uniform(0.01, 0.9)
        template_tokens = tokenizer.encode(TEMPLATE, add_special_tokens=True, return_tensors='pt').to(device)
        # randomly mask 25% of the template tokens, do not mask CLS or EOS tokens
        mask_index = torch.rand_like(template_tokens.float()) < mask_percentage
        mask_index[:, 0], mask_index[:, -1] = False, False
        template_tokens[mask_index] = tokenizer.mask_token_id

        # number of masked tokens
        steps = (template_tokens == tokenizer.mask_token_id).sum().item() // 10
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

        design = model._decode_seq(output_tokens[0])
        designs.append(design)

    designs.append(TEMPLATE)
    print(f'Number of designs: {len(designs)}')
    designs = list(set(designs))
    print(f'Number of unique designs: {len(designs)}')
    df = predict_against_target(target=TARGET, designs=designs)

    df = df.sort_values(by=['predicted-pKd'], ascending=True)
    # New column target-sites
    # column predicted-binding-sites is in the following format:
    # aminoacid-Location-chain-confidence
    # So something like A103a55 for alanine at position 103 on chain a with confidence 55
    # If all of the Target Aminos are in the predicted-binding-sites column with chain a, make True in target-sites column
    # drop the target column and rename SeqB to design
    df = df.drop(columns=['SeqA'])
    df = df.rename(columns={'SeqB': 'Design'})
    df['target-sites'] = df['predicted-binding-sites'].apply(lambda x: 'True' if all(f'{target_amino}a'.lower() in str(x).lower() for target_amino in TARGET_AMINOS) else False)

    # Where Design == TEMPLATE, make target-sites == 'TEMPLATE'
    df.loc[df['Design'] == TEMPLATE, 'target-sites'] = 'TEMPLATE'

    print(df.head())
    df.to_csv('designs.csv', index=False)
