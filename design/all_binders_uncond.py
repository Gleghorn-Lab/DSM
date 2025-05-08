import torch
import random
import argparse
import pandas as pd
import threading
import queue
import os
from tqdm import tqdm
from huggingface_hub import login, hf_hub_download
from safetensors.torch import load_file
from models.modeling_esm_diff import ESM_Diff_Binders, ESMDiffConfig
from models.utils import wrap_lora
from synthyra_api.affinity_pred import predict_against_target
from design.binder_info import BINDING_INFO


SYNTHYRA_API_KEY = '7147b8da62cc094c11d688dbac739e4689cdc7952d5196a488e5d95a6c2f2da1'
MODEL_PATH = 'GleghornLab/ESM_diff_650'
TEMPERATURE = 1.0
REMASKING = 'random'
SLOW = False
PREVIEW = True
STEP_DIVISOR = 100


### TODO
"""
There's an issue with how the api is getting called here

"""

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples per target')
    parser.add_argument('--test', action='store_true', help='Use test data instead of calling the API')
    parser.add_argument('--targets', nargs='+', default=None, help='Specific targets to design for')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--api_batch_size', type=int, default=25, help='API batch size')
    return parser.parse_args()


def prediction_worker(design_queue, result_queue, TARGET, args):
    """Worker function to process prediction batches in separate threads."""
    while True:
        batch_data = design_queue.get()
        if batch_data is None:  # Signal to terminate
            design_queue.task_done()
            break
            
        designs, batch_masks, target_name = batch_data
        print(f'Processing batch of {len(designs)} designs for target {target_name} in thread')
        
        # Get unique designs
        unique_designs = list(set(designs))
        print(f'Number of unique designs in batch: {len(unique_designs)}')
        
        # Predict against target
        batch_df = predict_against_target(target=TARGET, designs=unique_designs, test=args.test)
        
        # Add target name
        batch_df['target'] = target_name
        
        # Map mask rates to unique designs
        design_to_mask = {designs[i]: batch_masks[i] for i in range(len(designs))}
        batch_df['design_info'] = batch_df['SeqB'].map(design_to_mask)
        
        # Put result in queue
        result_queue.put(batch_df)
        design_queue.task_done()


def load_binder_model(model_path):
    local_weight_file = hf_hub_download(
        repo_id=model_path,
        filename='model.safetensors',
        repo_type='model',
    )

    config = ESMDiffConfig.from_pretrained(model_path)
    model = ESM_Diff_Binders(config=config)
    model = wrap_lora(model, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout)
    state_dict = load_file(local_weight_file)

    # Track which parameters were loaded
    loaded_params = set()
    missing_params = set()

    for name, param in model.named_parameters():
        found = False
        for key in state_dict.keys():
            if key in name:
                param.data = state_dict[key]
                loaded_params.add(name)
                found = True
                break
        if not found:
            missing_params.add(name)

    # Verify all weights were loaded correctly
    print(f"Loaded {len(loaded_params)} parameters")
    print(f"Missing {len(missing_params)} parameters")
    if missing_params:
        print("Missing parameters:")
        for param in sorted(missing_params):
            print(f"  - {param}")

    return model


def generate_designs_for_target(target_name, args, model, tokenizer, device, design_queue):
    """Generate designs for a specific target"""
    TARGET, TARGET_AMINOS, TARGET_IDX, TEMPLATE, TRUE_PKD, SOURCE, BINDER_SOURCE = BINDING_INFO[target_name]
    
    # Skip targets with incomplete information
    if not TARGET or not TEMPLATE or TRUE_PKD == 0.0:
        print(f"Skipping {target_name} due to incomplete information")
        return
    
    designs, design_info = [], []
    
    # Add template as first design
    designs.append(TEMPLATE)
    design_info.append(f'mask-rate: 0.0, positions: 0-{len(TEMPLATE)}')
    
    cls_token = tokenizer.cls_token_id
    eos_token = tokenizer.eos_token_id
    
    # Generate designs
    for sample in tqdm(range(args.num_samples // args.batch_size), desc=f"Generating designs for {target_name}"):
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
        
        target_tokens = tokenizer.encode(TARGET, add_special_tokens=True, return_tensors='pt').to(device)
        template_tokens = tokenizer.encode(template, add_special_tokens=False, return_tensors='pt').to(device)
        end_eos = torch.tensor([eos_token], device=device).unsqueeze(0)

        # expand template tokens to batch_size
        if args.batch_size > 1:
            target_tokens = target_tokens.repeat(args.batch_size, 1)
            template_tokens = template_tokens.repeat(args.batch_size, 1)
            end_eos = end_eos.repeat(args.batch_size, 1)

        # randomly mask template tokens
        mask_index = torch.rand_like(template_tokens.float()) < mask_percentage
        template_tokens[mask_index] = tokenizer.mask_token_id

        # cls, target, eos, template, eos
        template_tokens = torch.cat([target_tokens, template_tokens, end_eos], dim=1)

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
            batch_designs = [model._decode_seq(output_tokens[i])[len(TARGET):] for i in range(args.batch_size)]
            for design in batch_designs:
                designs.append(design)
                design_info.append(f'mask-rate: {round(mask_percentage, 2)}, positions: {start}-{end}')
        else:
            designs.append(model._decode_seq(output_tokens[0])[len(TARGET):])
            design_info.append(f'mask-rate: {round(mask_percentage, 2)}, positions: {start}-{end}')
        
        # Submit batch for processing when we reach api_batch_size
        if len(designs) >= args.api_batch_size:
            design_queue.put((designs.copy(), design_info.copy(), target_name))
            designs, design_info = [], []
    
    # Process any remaining designs
    if designs:
        design_queue.put((designs, design_info, target_name))


def summarize_results(all_dfs):
    """Analyze results and create summary dataframes"""
    if not all_dfs:
        print("No designs generated.")
        return
        
    # Concatenate all results
    results_df = pd.concat(all_dfs, ignore_index=True)
    
    # Get template info for each target
    template_info = {}
    target_info = {}
    for target, info in BINDING_INFO.items():
        if info[0] and info[3] and info[4] > 0:  # Check if target has valid data
            template_info[target] = {
                'template': info[3],
                'true_pKd': info[4]
            }
            target_info[target] = {
                'target_sequence': info[0],
                'target_residues': info[1],
                'binder_source': info[6]
            }
    
    # Rename columns for clarity
    results_df = results_df.rename(columns={
        'SeqA': 'Target_Sequence',
        'SeqB': 'Design',
        'predicted-pKd': 'Predicted_pKd'
    })
    
    # Add true pKd values for templates
    results_df['True_pKd'] = float('nan')
    for target, info in template_info.items():
        template_mask = (results_df['target'] == target) & (results_df['Design'] == info['template'])
        results_df.loc[template_mask, 'True_pKd'] = info['true_pKd']
        results_df.loc[template_mask, 'Is_Template'] = True
    
    results_df['Is_Template'] = results_df['Is_Template'].fillna(False)
    
    # Calculate accuracy metrics for templates
    template_df = results_df[results_df['Is_Template'] == True].copy()
    template_df['pKd_Error'] = template_df['Predicted_pKd'] - template_df['True_pKd']
    template_df['pKd_Error_Abs'] = template_df['pKd_Error'].abs()
    template_df['pKd_Error_Percent'] = (template_df['pKd_Error'] / template_df['True_pKd']) * 100
    
    # Create a summary dataframe for each target
    summary_data = []
    
    for target in results_df['target'].unique():
        if target not in template_info:
            continue
            
        target_df = results_df[results_df['target'] == target]
        template_row = target_df[target_df['Is_Template'] == True].iloc[0]
        true_pKd = template_row['True_pKd']
        predicted_template_pKd = template_row['Predicted_pKd']
        
        # Count designs with higher predicted pKd than template
        better_designs = target_df[(target_df['Predicted_pKd'] > predicted_template_pKd) & (target_df['Is_Template'] == False)]
        num_better_designs = len(better_designs)
        best_design_pKd = better_designs['Predicted_pKd'].max() if not better_designs.empty else predicted_template_pKd
        
        summary_data.append({
            'Target': target,
            'True_Template_pKd': true_pKd,
            'Predicted_Template_pKd': predicted_template_pKd,
            'pKd_Error': predicted_template_pKd - true_pKd,
            'pKd_Error_Percent': ((predicted_template_pKd - true_pKd) / true_pKd) * 100,
            'Total_Designs': len(target_df) - 1,  # Subtract the template
            'Better_Designs': num_better_designs,
            'Better_Designs_Percent': (num_better_designs / (len(target_df) - 1)) * 100 if len(target_df) > 1 else 0,
            'Best_Design_pKd': best_design_pKd,
            'pKd_Improvement': best_design_pKd - predicted_template_pKd,
            'Binder_Source': target_info[target]['binder_source'] if target in target_info else 'Unknown'
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create a dataframe with the best designs for each target
    best_designs_data = []
    
    for target in results_df['target'].unique():
        if target not in template_info:
            continue
            
        target_df = results_df[results_df['target'] == target]
        template_row = target_df[target_df['Is_Template'] == True].iloc[0]
        predicted_template_pKd = template_row['Predicted_pKd']
        
        # Get all designs with higher predicted pKd than template
        better_designs = target_df[(target_df['Predicted_pKd'] > predicted_template_pKd) & (target_df['Is_Template'] == False)]
        
        if not better_designs.empty:
            for _, row in better_designs.iterrows():
                best_designs_data.append({
                    'Target': target,
                    'Design': row['Design'],
                    'Predicted_pKd': row['Predicted_pKd'],
                    'Template_pKd': predicted_template_pKd,
                    'pKd_Improvement': row['Predicted_pKd'] - predicted_template_pKd,
                    'design_info': row['design_info'],
                    'binding_sites': row['predicted-binding-sites']
                })
    
    best_designs_df = pd.DataFrame(best_designs_data)
    if not best_designs_df.empty:
        best_designs_df = best_designs_df.sort_values(['Target', 'Predicted_pKd'], ascending=[True, False])
    
    return results_df, summary_df, best_designs_df


if __name__ == '__main__':
    # py -m design.multi_target_binder
    args = arg_parser()
    if args.token is not None:
        login(args.token)
    
    # Get list of available targets
    available_targets = []
    for target, info in BINDING_INFO.items():
        # Check if target has necessary information
        if info[0] and info[3] and info[4] > 0:  # Check for target sequence, template, and true pKd
            available_targets.append(target)
    
    # If specific targets are provided, validate them
    if args.targets:
        targets = [t for t in args.targets if t in available_targets]
        if not targets:
            print("None of the specified targets have complete information.")
            exit(1)
    else:
        targets = available_targets
    
    print(f"Processing {len(targets)} targets: {', '.join(targets)}")
    
    # Load model once
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_binder_model(MODEL_PATH)
    model = model.to(device).eval()
    tokenizer = model.tokenizer
    
    # Create queues for thread communication
    design_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Start worker threads
    num_threads = os.cpu_count() // 4  # Adjust based on your system capabilities
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=prediction_worker, args=(design_queue, result_queue, None, args))
        t.daemon = True
        t.start()
        threads.append(t)
    
    # Process each target
    for target_name in targets:
        TARGET = BINDING_INFO[target_name][0]
        # Update threads with current target
        for t in threads:
            if t.is_alive():
                t._args = (design_queue, result_queue, TARGET, args)
        
        generate_designs_for_target(target_name, args, model, tokenizer, device, design_queue)
    
    # Signal workers to terminate
    for _ in range(num_threads):
        design_queue.put(None)
    
    # Wait for all prediction tasks to complete
    design_queue.join()
    
    # Collect all dataframes
    all_dfs = []
    while not result_queue.empty():
        all_dfs.append(result_queue.get())
    
    # Summarize and save results
    results_df, summary_df, best_designs_df = summarize_results(all_dfs)
    
    # Save all results
    results_df.to_csv('all_designs.csv', index=False)
    summary_df.to_csv('design_summary.csv', index=False)
    best_designs_df.to_csv('better_designs.csv', index=False)
    
    print("\nDesign Summary:")
    print(summary_df.to_string(index=False))
    
    print("\nNumber of designs with improved predicted binding affinity:")
    for target in summary_df['Target'].unique():
        better = summary_df.loc[summary_df['Target'] == target, 'Better_Designs'].iloc[0]
        total = summary_df.loc[summary_df['Target'] == target, 'Total_Designs'].iloc[0]
        improvement = summary_df.loc[summary_df['Target'] == target, 'pKd_Improvement'].iloc[0]
        print(f"{target}: {better}/{total} designs, best improvement: {improvement:.2f} pKd") 