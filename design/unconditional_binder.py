import torch
import random
import argparse
import pandas as pd
import threading
import queue
import os
from tqdm import tqdm
from huggingface_hub import login
from models.modeling_esm_diff import ESM_Diff

from .binder_info import BINDING_INFO
from ..synthyra_api.affinity_pred import predict_against_target


MODEL_PATH = 'GleghornLab/ESM_diff_650'

TEMPERATURE = 1.0
REMASKING = 'random'
SLOW = False
PREVIEW = False
STEP_DIVISOR = 1


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--test', action='store_true', help='Use test data instead of calling the API')
    parser.add_argument('--target', type=str, default='EGFR', help='Target to design for')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--api_batch_size', type=int, default=25, help='API batch size')
    parser.add_argument('--synthyra_api_key', type=str, default=None, help='Synthyra API key')
    return parser.parse_args()


def prediction_worker(design_queue, result_queue, TARGET, args):
    """Worker function to process prediction batches in separate threads."""
    while True:
        batch_data = design_queue.get()
        if batch_data is None:  # Signal to terminate
            design_queue.task_done()
            break
            
        designs, batch_masks = batch_data
        print(f'Processing batch of {len(designs)} designs in thread')
        
        # Predict against target
        batch_df = predict_against_target(target=TARGET, designs=designs, test=args.test, api_key=args.synthyra_api_key)
        
        # Map mask rates to unique designs
        design_to_mask = {designs[i]: batch_masks[i] for i in range(len(designs))}
        batch_df['design_info'] = batch_df['SeqB'].map(design_to_mask)
        
        # Put result in queue
        result_queue.put(batch_df)
        design_queue.task_done()


if __name__ == '__main__':
    # py -m design.unconditional_binder
    args = arg_parser()
    if args.token is not None:
        login(args.token)
    
    TARGET, TARGET_AMINOS, TARGET_IDX, TEMPLATE, TRUE_PKD, SOURCE, BINDER_SOURCE = BINDING_INFO[args.target]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESM_Diff.from_pretrained(MODEL_PATH).to(device).eval()
    tokenizer = model.tokenizer

    designs, design_info, design_set = [], [], set()
    
    # Create queues for thread communication
    design_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Start worker threads
    num_threads = os.cpu_count() // 4  # Adjust based on your system capabilities
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=prediction_worker, args=(design_queue, result_queue, TARGET, args))
        t.daemon = True
        t.start()
        threads.append(t)
    
    designs.append(TEMPLATE)
    design_info.append(f'mask-rate: 0.0, positions: 0-{len(TEMPLATE)}')
    design_set.add(TEMPLATE)

    # Generate designs
    for sample in tqdm(range(args.num_samples // args.batch_size)):
        mask_percentage = random.uniform(0.01, 0.9)
        
        # Randomly select a region to mask
        if random.random() < 0.5:
            template_length = len(TEMPLATE)
            template_section_length = 0
            while template_section_length < template_length // 4:
                region_start = random.randint(0, template_length // 2)
                region_end = random.randint(region_start, template_length)
                template = TEMPLATE[region_start:region_end]
                start = region_start
                end = region_end
                template_section_length = end - start
        else:
            template = TEMPLATE
            start = 0
            end = len(TEMPLATE)
        template_tokens = tokenizer.encode(template, add_special_tokens=True, return_tensors='pt').to(device)
        # expand template tokens to batch_size
        if args.batch_size > 1:
            template_tokens = template_tokens.repeat(args.batch_size, 1)

        # randomly mask template tokens
        mask_index = torch.rand_like(template_tokens.float()) < mask_percentage
        mask_index[:, 0], mask_index[:, -1] = False, False
        template_tokens[mask_index] = tokenizer.mask_token_id

        # number of masked tokens
        steps = (template_tokens[0] == tokenizer.mask_token_id).sum().item() // STEP_DIVISOR
        output_tokens = model.mask_diffusion_generate(
            template_tokens=template_tokens,
            block_wise=False,
            steps=steps,
            temperature=TEMPERATURE,
            remasking=REMASKING,
            preview=PREVIEW,
            slow=SLOW,
            start_with_methionine=False
        )

        if args.batch_size > 1:
            batch_designs = [model._decode_seq(output_tokens[i]) for i in range(args.batch_size)]
            for design in batch_designs:
                if design in design_set:
                    continue
                designs.append(design)
                design_info.append(f'mask-rate: {round(mask_percentage, 2)}, positions: {start}-{end}')
        else:
            designs.append(model._decode_seq(output_tokens[0]))
            design_info.append(f'mask-rate: {round(mask_percentage, 2)}, positions: {start}-{end}')
        
        # Submit batch for processing when we reach api_batch_size
        if len(designs) >= args.api_batch_size:
            design_queue.put((designs.copy(), design_info.copy()))
            designs, design_info = [], []
    
    # Process any remaining designs
    if designs:
        design_queue.put((designs, design_info))
    
    # Signal workers to terminate
    for _ in range(num_threads):
        design_queue.put(None)
    
    # Wait for all prediction tasks to complete
    design_queue.join()
    
    # Collect all dataframes
    all_dfs = []
    while not result_queue.empty():
        all_dfs.append(result_queue.get())
    
    # Concatenate all dataframes
    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
        
        print(f'Total number of unique designs: {len(df)}')
        
        df = df.sort_values(by=['predicted-pKd'], ascending=False)

        # Drop the target column and rename SeqB to design
        df = df.drop(columns=['SeqA'])
        df = df.rename(columns={'SeqB': 'Design'})

        # Calculate target-sites
        #df['target-sites'] = df['predicted-binding-sites'].apply(
        #    lambda x: sum(1 for target_amino in TARGET_AMINOS if f'{target_amino}a'.lower() in str(x).lower())
        #)
        #df['target-sites'] = df['target-sites'].astype(object)

        ## Where Design == TEMPLATE, make target-sites == 'TEMPLATE'
        #df.loc[df['Design'] == TEMPLATE, 'target-sites'] = 'TEMPLATE'

        print(df.head())
        df.to_csv('designs.csv', index=False)
    else:
        print("No designs generated.")
