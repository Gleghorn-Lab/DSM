# DSM: Diffusion Models for Protein Sequence Generation
### Note: This readme is shared between our GitHub and Huggingface pages.

## Table of Contents
- [Introduction](#introduction)
- [Models](#models)
- [Usage](#usage)
- [Demos](#usage)
- [Local installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Cite](#cite)

## Introduction

DSM (Diffusion Sequence Model) is a novel Protein Language Model (pLM) developed in collaboration between the Gleghorn Lab and [Synthyra](https://synthyra.com/). It was trained with masked diffusion to enable both high-quality representation learning and generative protein design, detailed extensively in our [preprint](https://arxiv.org/abs/2506.08293). This repository contains the code for training and evaluating DSM and its variants.

DSM is capable of generating diverse, biomimetic sequences that align with expected amino acid compositions, secondary structures, and predicted functions, even under high corruption rates. Furthermore, DSM's learned representations match or exceed those of comparably sized pLMs on various downstream tasks. The repository also includes DSM-ppi, a variant fine-tuned to generate protein binders by attending to target sequences.

## Models

The following models are available on Hugging Face:

- **Base DSM Models**:
  - [GleghornLab/DSM_150](https://huggingface.co/GleghornLab/DSM_150) - 150M parameter DSM model
  - [GleghornLab/DSM_650](https://huggingface.co/GleghornLab/DSM_650) - 650M parameter DSM model

- **DSM-ppi Models**:
    (LoRA versions - results reported in paper but not recommended for real use)
  - [GleghornLab/DSM_150_ppi_lora](https://huggingface.co/GleghornLab/DSM_150_ppi_lora) - 150M parameter LoRA DSM-ppi model
  - [GleghornLab/DSM_650_ppi_lora](https://huggingface.co/GleghornLab/DSM_650_ppi_Lora) - 650M parameter LoRA DSM-ppi model
  - [GleghornLab/DSM_150_ppi_control](https://huggingface.co/GleghornLab/DSM_150_ppi_control) - Control version of LoRA DSM-ppi
    (Fully finetuned - recommended for real use)
  - [Synthyra/DSM_ppi_full](https://huggingface.co/Synthyra/DSM_ppi_full) - 650M parameter DSM-ppi model

- **Datasets**:
  - [Synthyra/omg_prot50](https://huggingface.co/Synthyra/omg_prot50) - Open MetaGenomic dataset clustered at 50% identity (207M sequences)
  - [GleghornLab/stringv12_modelorgs_9090](https://huggingface.co/GleghornLab/stringv12_modelorgs_9090) - STRING database model organisms (653k sequences)

- **Utility Models**:
  - [GleghornLab/production_ss4_model](https://huggingface.co/GleghornLab/production_ss4_model) - Secondary structure prediction (4-class)
  - [GleghornLab/production_ss9_model](https://huggingface.co/GleghornLab/production_ss9_model) - Secondary structure prediction (9-class)

## Usage

This section outlines how to use a trained `DSM` model for common generation tasks. The core generation logic is provided by the `GenerateMixin` class, used by `DSM` models.

First, ensure you have a trained model (either one you trained or a pre-trained one from Hugging Face Hub) and the necessary environment set up.

```python
import torch
from models.modeling_dsm import DSM # Or DSM_ppi for binder generation

# Load a pre-trained model
model_name_or_path = "GleghornLab/DSM_650" # Replace with your model of choice
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DSM.from_pretrained(model_name_or_path).to(device).eval()
tokenizer = model.tokenizer
```

```console
You are using a model of type esm_diff to instantiate a model of type dsm. This is not supported for all configurations of models and can yield errors.
```
This warning is normal - all good!

### 1. Unconditional Sequence Generation
To generate a novel sequence of a specific length. DSM uses a progressive denoising approach.

```python
### Unconditional generation
length = 100
mask_token = tokenizer.mask_token
# optionally, enforce starting with methionine
input_template = tokenizer.encode('M' + ''.join([mask_token] * (length - 1)), add_special_tokens=True).to(device)
output = model.mask_diffusion_generate(
    input_tokens=input_template,
    step_divisor=100,   # lower is slower but better
    temperature=1.0,    # sampling temperature
    remasking="random", # strategy for remasking tokens not kept
    preview=False       #
)

generated_sequences = model.decode_output(output)
print(f"Generated sequence: {generated_sequences[0]}")
```

```console
Generated sequence: MFRVDALQVAQQETLAIGRSTAYDKQESPSMAQRQVLTQLAAYGGENDLRQICIPAERRNFLSIANGASYQFVEEDNEANGGYWSPHKAGLPESACKRFI
```

### 2. Mask Filling (Inpainting)
To fill in masked regions of a template sequence:

```python
# Mask Filling / Inpainting
template_sequence = "MA<mask><mask><mask>KEG<mask><mask>STL"
template_tokens = model.tokenizer.encode(template_sequence, add_special_tokens=True).to(device)

filled_ids = model.mask_diffusion_generate(
    input_tokens=template_tokens,
    step_divisor=100,   # lower is slower but better
    temperature=1.0,    # sampling temperature
    remasking="random", # strategy for remasking tokens not kept
    preview=False
)

generated_sequences = model.decode_output(output)
print(f"Generated sequence: {generated_sequences[0]}")
```

```console
Generated sequence: MAVKFKEGGISTL
```

### 3. Conditional Generation (e.g., Binders - using DSM-ppi)
If using DSM-ppi, the input format is specific for generating a binder (SeqB) for a target (SeqA).

```python
# from models.modeling_dsm import DSM_ppi
# model_binder = DSM_ppi.from_pretrained("GleghornLab/DSM_650_ppi_lora").to(device).eval()
# The lora version from the paper leads to unreliable outputs
# Synthyra has generously trained a version through full fine tuning
from models.modeling_dsm import DSM

model_binder = DSM.from_pretrained("Synthyra/DSM_ppi_full").to(device).eval()

target_seq = "TARGETSEQUENCEAMINOACIDS"
# For binder generation, the 'interactor' (SeqB) part is what gets generated/filled.
# Start with a fully masked interactor of desired length.
interactor_template_len = 20
interactor_template = ''.join([mask_token] * interactor_template_len)

combined_input_str = target_seq + '<eos>' + interactor_template

binder_template_tokens = tokenizer.encode(combined_input_str, add_special_tokens=True).to(device)

output = model_binder.mask_diffusion_generate(
    input_tokens=binder_template_tokens,
    step_divisor=100,   # lower is slower but better
    temperature=1.0,    # sampling temperature
    remasking="random", # strategy for remasking tokens not kept
)

target, binder = model.decode_dual_input(output, seperator='<eos>;)
# Parse out the generated interactor part based on EOS tokens.
# Example: generated_full_seq_str.split(model_binder.tokenizer.eos_token)[1]
print(f"Generated binder {binder[0]}")
```

```console
Generated binder HRHHHRRPTHARETEWLARMRLGIAEHQRIAVPRSDLEPDQMRERAADNQRLVKEYDQVIDHQTEGSTERLFEVLRVWEQVNTEQAHHEASAALEFGRVGYPDDEGGRAFYTQANAHKKDLVEYIGGIDEDAKWDPRIAWLMPEGGQPVKATVIGVSEERINGLKVLDDHWGRERRLWLINLFTALQAYDDPTRPTQVTLTPATDQLTNDVQYLLLSTRYTPPGVTTAVKIRKLDGRTLKVLTTEAPYVVRGATLS
```
Folded with Chai1:

![image](https://github.com/user-attachments/assets/782d7bba-6f25-4a27-b0c4-fef88565dd33)


## Demos
There are various demos with many more to come. For example, in `demo_dsm_ppi_full.py` (run by `python -m demos.demo_dsm_ppi_full`) we perform a test on DSM-ppi.
We take 1000 proteins pairs from BIOGRID (real protein-protein interactions) and 1000 from Negatome (non interacting protein pairs) and mask the second sequence (SeqB) by 50%.
This acts as a sanity check, as we expect the accuracy on reconstructing real positive PPIs to be higher than the accuracy on non-interacting proteins.
Indeed, this is the case:

```console
==================================================
RESULTS COMPARISON
==================================================
Positive examples:
  Mean accuracy: 0.495 ± 0.322
  Processed:     1000 examples

Negative examples:
  Mean accuracy: 0.227 ± 0.231
  Processed:     1000 examples

Difference (Positive - Negative): 0.267
T-test: t=21.331, p=0.000
Difference is statistically significant (p < 0.05)
``` 


## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Set up the Python virtual environment:**
    The `setup_bioenv.sh` script creates a virtual environment named `bioenv` in your home directory (`~/bioenv`), installs PyTorch with CUDA 12.6 support, and then installs all other dependencies from `requirements.txt`.

    Make the script executable:
    ```bash
    chmod +x setup_bioenv.sh
    ```
    Run the script:
    ```bash
    ./setup_bioenv.sh
    ```

3.  **Activate the environment:**
    Each time you want to work on this project, activate the virtual environment:
    ```bash
    source ~/bioenv/bin/activate
    ```

4.  **To deactivate the environment:**
    ```bash
    deactivate
    ```

## Training

The primary script for training models is `training/train_dsm.py`. This script further pretrains an ESM2 checkpoint using the DSM objective (masked diffusion based on LLaDA) on a large protein sequence dataset like [OMG-prot50](https://huggingface.co/Synthyra/omg_prot50).

### Main Training Script: `train_dsm.py`

-   **Base Model**: DSM models are extended from pre-trained ESM2 checkpoints (e.g., ESM2-150M, ESM2-650M).
-   **Training Objective**: Masked diffusion loss, where the model predicts masked tokens. The loss is scaled by `1/(t + epsilon)` where `t` is the corruption level, penalizing errors more at low mask rates.
-   **Language Modeling Head**: Uses a modified head with a soft-logit cap (`tau=30`) and tied output projection weights to the token embeddings.
-   **Data Handling**:
    -   Training data can be streamed from datasets like [Synthyra/omg_prot50](https://huggingface.co/Synthyra/omg_prot50) (a version of Open MetaGenomic dataset clustered at 50% identity).
    -   Uses `data.dataset_classes.SequenceDatasetFromList` for validation/test sets and `data.dataset_classes.IterableDatasetFromHF` for streaming training.
    -   `data.data_collators.SequenceCollator` is used for batching.
-   **Training Process**:
    -   Utilizes Hugging Face `TrainingArguments`.
    -   A custom `IterableTrainer` (from `training.iterable_trainer.py`) handles iterable datasets.
    -   Uses AdamW optimizer and a cosine learning rate scheduler with linear warmup.
    -   Supports logging to Weights & Biases (wandb).
    -   The trained model can be pushed to Hugging Face Hub.
    -   Example checkpoints mentioned in the paper: [DSM-150](https://huggingface.co/GleghornLab/DSM_150) (from ESM2-150M, 100k steps, batch 32, seqlen 512, LR 1e-4) and [DSM-650](https://huggingface.co/GleghornLab/DSM_650) (from ESM2-650M, 100k steps, global batch 128, seqlen 2048, LR 1e-4).

**Usage Example:**

```bash
python -m training.train_dsm \
    --model_path facebook/esm2_t33_650M_UR50D \
    --save_path GleghornLab/DSM_650 \
    --lr 1e-4 \
    --batch_size 8 \
    --grad_accum 16 \
    --max_steps 100000 \
    --save_every 1000 \
    --fp16 \
    --wandb_project "DSM_Training" \
    --token <your_hf_token_if_needed_for_private_repo_or_saving>
```

**Key Command-Line Arguments for `train_dsm.py`:**

*   `--token`: Hugging Face token.
*   `--model_path`: Path to the base ESM2 model to start from.
*   `--save_path`: Path to save the trained DSM model on Hugging Face Hub.
*   `--lr`: Learning rate.
*   `--batch_size`: Batch size per device.
*   `--grad_accum`: Gradient accumulation steps.
*   `--max_steps`: Maximum training steps.
*   `--wandb_project`: Wandb project name (default: `DSM`).
*   `--max_length`: Maximum sequence length.
*   `--save_every`: Save model and evaluate every N steps.
*   `--fp16`: Enable mixed-precision training.
*   `--bugfix`: Use small batch size and max length for debugging.

### Other Training Scripts (e.g., for DSM-ppi)

The `training/` directory may also contain scripts like `train_dsm_bind.py`.
-   DSM-ppi (e.g., [DSM-150-ppi](https://huggingface.co/GleghornLab/DSM_150_ppi), [DSM-650-ppi](https://huggingface.co/GleghornLab/DSM_650_ppi)) is fine-tuned on PPI datasets.
-   Training involves conditioning on a target sequence (SeqA) to generate an interactor (SeqB) using the format `[CLS]--SeqA--[EOS]--[MASKED~SeqB]--[EOS]`.
-   LoRA (Low-Rank Adaptation) can be applied to attention layers for efficient fine-tuning.

And `training/iterable_trainer.py` provides the `get_iterable_trainer` function used by `train_dsm.py` to enable training with iterable datasets.

## Evaluation

The repository includes a comprehensive suite for evaluating model performance, focusing on:

1.  **Sequence Reconstruction (Mask Filling):**
    *   Evaluated by masking validation/test sets at various corruption rates (5% to 90%) and measuring cross-entropy loss, weighted F1 score, and Alignment Score (ASc) for the masked positions.
    *   The script `evaluation/mask_filling.py` is central to this.

2.  **Unconditional Generation Quality:**
    *   Generate a corpus of sequences based on lengths from a reference set (e.g., validation data).
    *   Compare distributions (1-mers, 2-mers, 3-mers) of amino acids and predicted secondary structures between generated and natural sequences using χ² test and Jensen-Shannon (JS) divergence.
    *   Compare distributions of predicted functional annotations (e.g., using Annotation Vocabulary - AV terms).
    *   Scripts involved: `evaluation/unconditional_generation_tuning.py` (to find optimal generation parameters like temperature and step divisor `s`), `evaluation/unconditional_generation.py`, `evaluation/ss_pred.py` (using [production_ss4_model](https://huggingface.co/GleghornLab/production_ss4_model) or [production_ss9_model](https://huggingface.co/GleghornLab/production_ss9_model)), `evaluation/annotate_comparisons.py`, `evaluation/compare_distributions.py`, `evaluation/plot_distribution_comparisons.py`.
    *   The `run_eval_pipeline.py` script automates this workflow.

3.  **Representation Quality (Model Probing):**
    *   Evaluate learned embeddings by training linear probes (or simple transformer blocks) on various downstream tasks (e.g., secondary structure prediction, localization prediction, etc.).
    *   Performance is compared against random vectors, randomized transformers, and other established pLMs.

4.  **Conditional Generation (Binder Design for DSM-ppi):**
    *   Evaluate DSM-ppi on benchmarks like BenchBB.
    *   Generate binders for target proteins using template-based masking strategies.
    *   Assess generated binders using *in-silico* tools like Synteract2 for predicted binding affinity (ppKd).

The `evaluation/` directory also contains a `readme.md` which provides further details on some evaluation workflows. Key metrics used include:
-   **Alignment Score (ASc):** A normalized Needleman-Wunsch global alignment score (using BLOSUM62) to measure sequence similarity, robust to length variations. ASc(a, b) = l/(f(a, a) - f(a, b) + l).
-   **Jensen-Shannon (JS) Divergence:** To compare distributions of k-mers and functional terms.

**Running the Full Unconditional Evaluation Pipeline:**

```bash
python run_eval_pipeline.py --token YOUR_HF_TOKEN --data_dir ./evaluation_results
```

Refer to `run_eval_pipeline.py --help` for more options, such as `--skip_tuning`.

### Mask Filling Evaluation

The script `evaluation/mask_filling.py` is used to evaluate models on their ability to predict masked tokens in a sequence across various masking rates.

-   **Functionality:**
    -   Evaluates different models (DSM, DPLM, standard ESM models).
    -   Tests across multiple datasets ([Synthyra/omg_prot50](https://huggingface.co/Synthyra/omg_prot50), [GleghornLab/stringv12_modelorgs_9090](https://huggingface.co/GleghornLab/stringv12_modelorgs_9090)).
    -   Calculates metrics: loss, perplexity, precision, recall, F1, accuracy, MCC, and alignment score.
    -   Saves detailed results to CSV files.
    -   Can generate a summary plot comparing model performance across different mask rates using `evaluation/plot_mask_fill_results.py`.

-   **Usage Example:**
    ```bash
    python -m evaluation.mask_filling \
        --token YOUR_HF_TOKEN \
        --batch_size 4 \
        --mask_rates 0.15 0.30 0.50 \
        --data_splits valid test \
        --results_dir ./results/mask_fill_custom
    ```
    To generate a comparison plot from existing results:
    ```bash
    python -m evaluation.mask_filling --generate_comparison_plot --results_dir ./results/mask_fill_custom --plot_output ./results/mask_fill_custom/comparison.png
    ```

### Other Evaluation Scripts

The `evaluation/` directory contains additional scripts for more specific analyses. These are typically run independently:

-   `evaluation/all_targets_uncond.py` and `evaluation/all_targets_cond.py`: Likely for evaluating generation towards specific targets, unconditionally and conditionally.
-   `evaluation/conditional_binder.py` and `evaluation/unconditional_binder.py`: Suggest evaluation focused on generating protein binders.
-   `evaluation/unconditional_by_length.py`: May evaluate unconditional generation focusing on sequence length distributions.
-   `evaluation/utils.py`: Utility functions for evaluation scripts.

Users should refer to individual scripts (e.g., using `python -m evaluation.<script_name> --help`) for their specific usage and arguments.
The `evaluation/` directory also contains a `readme.md` which provides further details on the unconditional generation evaluation workflow.

## Results

DSM demonstrates strong performance in both protein sequence generation and representation learning, establishing masked diffusion as a powerful paradigm.

-   **Biomimetic Sequence Generation**: Unconditionally generated DSM sequences closely mimic natural protein distributions in terms of amino acid k-mers, predicted secondary structures (JS divergence < 0.01 for AA k-mers), and predicted functional annotations (AV terms, JS divergence ~0.1). This suggests DSM captures underlying biological principles.

-   **Superior Sequence Reconstruction**: DSM models significantly outperform MLM-based ESM2 models in reconstructing sequences from highly corrupted inputs (up to 90% masking).
    -   At 90% masking, DSM achieves an Alignment Score (ASc) of ~0.27, considerably higher than random.
    -   DSM models show higher F1 scores in reconstruction tasks compared to DPLM models, especially at high mask rates.

-   **High-Quality Embeddings**: DSM embeddings match or exceed the quality of those from comparably sized pLMs (ESM2, DPLM) and even larger autoregressive models (ProtCLM 1B) on various downstream tasks evaluated by linear probing. [DSM-650](https://huggingface.co/GleghornLab/DSM_650) generally provides the best representations among tested models of similar size.

-   **Effective Binder Design (DSM-ppi):**
    -   [DSM-ppi](https://huggingface.co/GleghornLab/DSM_150_ppi) fine-tuned on protein-protein interaction data, demonstrates the ability to generate protein binders conditioned on target sequences.
    -   On the BenchBB benchmark, DSM-generated binders (both unconditional DSM and conditional DSM-ppi) show promising predicted binding affinities, in some cases superior to known binders. For example, designs for EGFR showed high predicted pKd and good structural metrics (ipTM, pTM with AlphaFold3).

-   **Efficiency**: DSM can generate realistic protein sequences from a single forward pass during reconstruction tasks at high mask rates, offering potential efficiency advantages over iterative AR or some discrete diffusion models.

These results highlight DSM's capability to unify high-quality protein representation learning and biologically coherent generative modeling within a single framework.

## Cite
```
@misc{hallee2025diffusionsequencemodelsenhanced,
      title={Diffusion Sequence Models for Enhanced Protein Representation and Generation}, 
      author={Logan Hallee and Nikolaos Rafailidis and David B. Bichara and Jason P. Gleghorn},
      year={2025},
      eprint={2506.08293},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2506.08293}, 
}
```
