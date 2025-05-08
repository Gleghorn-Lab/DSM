import torch
import torch.nn as nn
import pandas as pd
import os
from datasets import load_dataset
from peft import LoraConfig, LoraModel
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, login
from transformers import AutoModelForTokenClassification
from tqdm.auto import tqdm


SS4_MODEL_PATH = 'lhallee/SS3_ESMC-600_2025-05-07-21-24_XKSX'
SS9_MODEL_PATH = 'lhallee/SS8_ESMC-600_2025-05-07-21-24_XKSX'


def wrap_lora(module: nn.Module, r: int, lora_alpha: float, lora_dropout: float) -> nn.Module:
    # these modules handle ESM++ and ESM2 attention types, as well as any additional transformer blocks from Syndev
    target_modules=["layernorm_qkv.1", "out_proj", "query", "key", "value", "dense"]
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    module = LoraModel(module, lora_config, 'default')
    for name, param in module.named_parameters():
        if 'classifier' in name.lower():
            param.requires_grad = True
    return module


def tag_to_id():
    ss4_data = load_dataset('GleghornLab/SS3')
    ss9_data = load_dataset('GleghornLab/SS8')

    train_labels = ss4_data['train']['labels']
    unique_tags = set(tag for doc in train_labels for tag in doc)
    id2tag_ss4 = {id: tag for id, tag in enumerate(sorted(unique_tags))}
    
    train_labels = ss9_data['train']['labels']
    unique_tags = set(tag for doc in train_labels for tag in doc)
    id2tag_ss9 = {id: tag for id, tag in enumerate(sorted(unique_tags))}

    return id2tag_ss4, id2tag_ss9


def load_ss_model(model_path):
    """
    This will throw a bunch of warnings
    Because we load manually after the weights are the right shape, the model is correct
    """
    local_weight_file = hf_hub_download(
        repo_id=model_path,
        filename='model.safetensors',
        repo_type='model',
    )

    num_labels = 4 if 'ss3' in model_path.lower() else 9
    model = AutoModelForTokenClassification.from_pretrained(model_path, trust_remote_code=True, num_labels=num_labels)
    model = wrap_lora(model, r=64, lora_alpha=32, lora_dropout=0.01)
    state_dict = load_file(local_weight_file)

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

    model.eval()
    return model
    

if __name__ == '__main__':
    # py -m evaluation.ss_pred
    import argparse

    def arg_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--token', type=str, default=None)
        parser.add_argument('--input_path', type=str, default='test_compare.csv')
        parser.add_argument('--output_path', type=str, default='test_compare_ss_pred.csv')
        return parser.parse_args()

    args = arg_parser()
    if args.token is not None:
        login(args.token)

    args.input_path = os.path.join('evaluation', 'comparisons', args.input_path)
    args.output_path = os.path.join('evaluation', 'comparisons', args.output_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(args.input_path)
    natural_seqs = df['natural'].tolist()
    generated_seqs = df['generated'].tolist()

    id2tag_ss4, id2tag_ss9 = tag_to_id()
    ss4_model = load_ss_model(SS4_MODEL_PATH)
    ss9_model = load_ss_model(SS9_MODEL_PATH)

    # Get tokenizers for both models
    tokenizer = ss4_model.tokenizer

    nat_ss4_preds, nat_ss9_preds, gen_ss4_preds, gen_ss9_preds = [], [], [], []

    ss4_model = ss4_model.to(device)

    for seq in tqdm(natural_seqs, desc="Natural sequences - SS4"):
        tokens = tokenizer(seq, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = ss4_model(**tokens).logits
            preds = torch.argmax(logits, dim=-1).flatten().cpu().tolist()
            pred_letters = [id2tag_ss4[pred] for pred in preds]
            final_pred = ''.join(pred_letters)
            nat_ss4_preds.append(final_pred)

    for seq in tqdm(generated_seqs, desc="Generated sequences - SS4"):
        tokens = tokenizer(seq, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = ss4_model(**tokens).logits
            preds = torch.argmax(logits, dim=-1).flatten().cpu().tolist()
            pred_letters = [id2tag_ss4[pred] for pred in preds]
            final_pred = ''.join(pred_letters)
            gen_ss4_preds.append(final_pred)

    ss4_model.cpu()
    del ss4_model
    torch.cuda.empty_cache()

    ss9_model = ss9_model.to(device)

    for seq in tqdm(natural_seqs, desc="Natural sequences - SS9"):
        tokens = tokenizer(seq, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = ss9_model(**tokens).logits
            preds = torch.argmax(logits, dim=-1).flatten().cpu().tolist()
            pred_letters = [id2tag_ss9[pred] for pred in preds]
            final_pred = ''.join(pred_letters)
            nat_ss9_preds.append(final_pred)

    for seq in tqdm(generated_seqs, desc="Generated sequences - SS9"):
        tokens = tokenizer(seq, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = ss9_model(**tokens).logits
            preds = torch.argmax(logits, dim=-1).flatten().cpu().tolist()
            pred_letters = [id2tag_ss9[pred] for pred in preds]
            final_pred = ''.join(pred_letters)
            gen_ss9_preds.append(final_pred)

    ss9_model.cpu()
    del ss9_model
    torch.cuda.empty_cache()

    # Add all predictions to dataframe
    df['nat_ss4'] = nat_ss4_preds
    df['nat_ss9'] = nat_ss9_preds
    df['gen_ss4'] = gen_ss4_preds
    df['gen_ss9'] = gen_ss9_preds
    
    # Save results
    df.to_csv(args.output_path, index=False)
