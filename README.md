# DSM: Diffusion Models for Protein Sequence Generation

DSM (Diffusion Sequence Model) is a novel Protein Language Model (pLM) trained with masked diffusion to enable both high-quality representation learning and generative protein design. This repository contains the code for training and evaluating DSM and its variants. DSM builds upon the ESM2 architecture by incorporating a masked forward diffusion process inspired by the LLaDA framework.

DSM is capable of generating diverse, biomimetic sequences that align with expected amino acid compositions, secondary structures, and predicted functions, even under high corruption rates. Furthermore, DSM's learned representations match or exceed those of comparably sized pLMs on various downstream tasks. The repository also includes DSM$_{ppi}$, a variant fine-tuned to generate protein binders by attending to target sequences.

The repository provides scripts for:
-   Training `DSM` models (e.g., DSM$_{150}$, DSM$_{650}$) and its variants like DSM$_{ppi}$.
-   Comprehensive evaluation of unconditional generation, sequence reconstruction (mask filling), and representation quality.
-   Generating protein sequences using diffusion and autoregressive methods.

## Table of Contents
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Functionality](#functionality)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)

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

## Dependencies

The project relies on the following major libraries. All dependencies are listed in `requirements.txt` and installed by the `setup_bioenv.sh` script.

-   torch>=2.6.0
-   torchvision
-   torchmetrics
-   torchaudio
-   torchinfo
-   tf-keras
-   tensorflow
-   transformers>=4.48
-   accelerate>=1.1.0
-   datasets
-   einops
-   numpy==1.26.1
-   pandas
-   scikit-learn
-   scipy
-   wandb
-   peft
-   matplotlib
-   biopython
-   biotite
-   seaborn
-   pauc
-   sentencepiece
-   IPython
-   wordcloud

## Functionality

The core models in this repository, `DSM` and its variants (defined in `models/modeling_dsm.py`), offer several functionalities related to protein sequence generation:

1.  **Unconditional Protein Sequence Generation:**
    *   Models can generate novel protein sequences from scratch using the `mask_diffusion_generate` method (from `models.generate_mixin.py`).
    *   This involves an iterative denoising process, inspired by diffusion models, starting from a fully masked sequence. DSM is trained using a masked forward diffusion scheme based on the LLaDA framework.

2.  **Mask Filling / Sequence Reconstruction:**
    *   Given a protein sequence with masked regions (template), the models can fill in the missing residues with high fidelity, even at high corruption rates (e.g., 90% masking).
    *   This also uses the `mask_diffusion_generate` method.

3.  **Conditional Generation (Protein Binders):**
    *   The `DSM_{ppi}` model variant is specifically fine-tuned for protein binder generation. It takes a target sequence as input and generates a potential interacting sequence (binder).
    *   The input format for `DSM_{ppi}` during training and generation is typically `[CLS]--SeqA--[EOS]--[MASKED~SeqB]--[EOS]`.
    *   The `mask_diffusion_generate` method in general also includes a `prompt_tokens` argument, suggesting broader capabilities for conditional generation.

4.  **High-Quality Protein Representations:**
    *   DSM learns rich, semantic representations of protein sequences. These embeddings match or outperform those from comparably sized pLMs (including ESM2 and DPLM) on a variety of downstream tasks, as evaluated by linear probing.

5.  **Autoregressive Generation:**
    *   Models inheriting `GenerateMixin` also support standard left-to-right autoregressive generation via the `auto_regressive_generate` method, useful for comparison or as an alternative generation strategy.

DSM models feature a modified language modeling head with a soft-logit cap (scaled tanh activation) to stabilize training and improve sampling quality. The primary generation mechanism, `mask_diffusion_generate`, allows for flexible control over the generation process.

## Training

The primary script for training models is `training/train_dsm.py`. This script further pretrains an ESM2 checkpoint using the DSM objective (masked diffusion based on LLaDA) on a large protein sequence dataset like OMG$_{prot50}$.

### Main Training Script: `train_dsm.py`

-   **Base Model**: DSM models are extended from pre-trained ESM2 checkpoints (e.g., ESM2$_{150M}$, ESM2$_{650M}$).
-   **Training Objective**: Masked diffusion loss, where the model predicts masked tokens. The loss is scaled by `1/(t + epsilon)` where `t` is the corruption level, penalizing errors more at low mask rates.
-   **Language Modeling Head**: Uses a modified head with a soft-logit cap (`tau=30`) and tied output projection weights to the token embeddings.
-   **Data Handling**:
    -   Training data can be streamed from datasets like `Synthyra/omg_prot50` (a version of Open MetaGenomic dataset clustered at 50% identity).
    -   Uses `data.dataset_classes.SequenceDatasetFromList` for validation/test sets and `data.dataset_classes.IterableDatasetFromHF` for streaming training.
    -   `data.data_collators.SequenceCollator` is used for batching.
-   **Training Process**:
    -   Utilizes Hugging Face `TrainingArguments`.
    -   A custom `IterableTrainer` (from `training.iterable_trainer.py`) handles iterable datasets.
    -   Uses AdamW optimizer and a cosine learning rate scheduler with linear warmup.
    -   Supports logging to Weights & Biases (wandb).
    -   The trained model can be pushed to Hugging Face Hub.
    -   Example checkpoints mentioned in the paper: DSM$_{150}$ (from ESM2$_{150M}$, 100k steps, batch 32, seqlen 512, LR 1e-4) and DSM$_{650}$ (from ESM2$_{650M}$, 100k steps, global batch 128, seqlen 2048, LR 1e-4).

**Usage Example:**

```bash
python -m training.train_dsm \\
    --model_path facebook/esm2_t33_650M_UR50D \\
    --save_path YourHuggingFaceUser/DSM_650M_finetuned \\
    --lr 1e-4 \\
    --batch_size 8 \\
    --grad_accum 16 \\
    --max_steps 100000 \\
    --save_every 1000 \\
    --fp16 \\
    --wandb_project "DSM_Training" \\
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

### Other Training Scripts (e.g., for `DSM_{ppi}`)

The `training/` directory may also contain scripts like `train_dsm_bind.py`.
-   `DSM_{ppi}` (e.g., DSM$_{150-ppi}$, DSM$_{650-ppi}$) is fine-tuned on PPI datasets.
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
    *   Compare distributions (1-mers, 2-mers, 3-mers) of amino acids and predicted secondary structures between generated and natural sequences using $\chi^2$ test and Jensen-Shannon (JS) divergence.
    *   Compare distributions of predicted functional annotations (e.g., using Annotation Vocabulary - AV terms).
    *   Scripts involved: `evaluation/unconditional_generation_tuning.py` (to find optimal generation parameters like temperature and step divisor `s`), `evaluation/unconditional_generation.py`, `evaluation/ss_pred.py`, `evaluation/annotate_comparisons.py`, `evaluation/compare_distributions.py`, `evaluation/plot_distribution_comparisons.py`.
    *   The `run_eval_pipeline.py` script automates this workflow.

3.  **Representation Quality (Model Probing):**
    *   Evaluate learned embeddings by training linear probes (or simple transformer blocks) on various downstream tasks (e.g., secondary structure prediction, localization prediction, etc.).
    *   Performance is compared against random vectors, randomized transformers, and other established pLMs.

4.  **Conditional Generation (Binder Design for `DSM_{ppi}`):**
    *   Evaluate `DSM_{ppi}` on benchmarks like BenchBB.
    *   Generate binders for target proteins using template-based masking strategies.
    *   Assess generated binders using \textit{in-silico} tools like Synteract2 for predicted binding affinity (ppKd).

The `evaluation/` directory also contains a `readme.md` which provides further details on some evaluation workflows. Key metrics used include:
-   **Alignment Score (ASc):** A normalized Needleman-Wunsch global alignment score (using BLOSUM62) to measure sequence similarity, robust to length variations. $ASc(a, b) = \frac{l}{f(a, a) - f(a, b) + l}$.
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
    -   Tests across multiple datasets (`Synthyra/omg_prot50`, `lhallee/string_model_org_90_90_split`).
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

## Usage

This section outlines how to use a trained `DSM` model for common generation tasks. The core generation logic is provided by the `GenerateMixin` class, used by `DSM` models.

First, ensure you have a trained model (either one you trained or a pre-trained one from Hugging Face Hub) and the necessary environment set up.

```python
import torch
from models.modeling_dsm import DSM # Or DSM_ppi for binder generation
# from transformers import AutoTokenizer # Tokenizer is usually part of the DSM model

# Load a pre-trained model
model_name_or_path = "YourHuggingFaceUser/DSM_650M_finetuned" # Replace with your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DSM.from_pretrained(model_name_or_path).to(device).eval()
# model.tokenizer is available
```

### 1. Unconditional Sequence Generation
To generate a novel sequence of a specific length. DSM uses a progressive denoising approach.

```python
# Unconditional generation
generated_ids = model.mask_diffusion_generate(
    batch_size=1,
    length=100,  # Desired length of the protein (excluding CLS/EOS)
    steps=5, # Number of diffusion steps (s value from paper, e.g. length // 5 for 5 tokens per step, or a fixed number of steps)
                # The paper mentions s=5 (step divisor) meaning L/5 steps, or filling L/s tokens per step.
                # Or simpler: steps = desired number of iterations, e.g., 20-100. Let's use a fixed number.
    # For full diffusion, steps might be ~num_tokens_to_generate / tokens_per_step
    # The paper mentions "s=5 for future experimentation" where s is a step divisor.
    # And "filling in one token at a time (s=1)" or "s tokens were chosen to keep".
    # Let's use 'steps' as the number of iterations for clarity here.
    steps=int(100 / 5), # Example: if length is 100 and we unmask 5 tokens effectively per step
    temperature=1.0,    # Sampling temperature
    remasking="random", # Strategy for remasking tokens not kept
    start_with_methionine=True,
    preview=False
)

generated_sequence = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Generated sequence: {generated_sequence.replace(' ', '')}")
```

### 2. Mask Filling (Inpainting)
To fill in masked regions of a template sequence:

```python
# Mask Filling / Inpainting
template_sequence = "MA<MASK><MASK><MASK>KEG<MASK><MASK>STL"
# Ensure template includes CLS and EOS if model expects them for processing, 
# or handle tokenization appropriately. The GenerateMixin adds them if not template_tokens.
# For direct use with mask_diffusion_generate, provide tokenized template_tokens.

# Tokenize the template (model.tokenizer can be used)
# Example assumes CLS and EOS are handled by the model or GenerateMixin
input_str_for_tokenizer = f"{model.tokenizer.cls_token}{template_sequence}{model.tokenizer.eos_token}"

template_tokens = model.tokenizer.encode(input_str_for_tokenizer, return_tensors="pt").to(device)

filled_ids = model.mask_diffusion_generate(
    template_tokens=template_tokens,
    steps=int(template_tokens.shape[1] / 5), # Adjust steps based on sequence length or number of masks
    temperature=0.8,
    remasking="low_confidence",
    preview=False
)

filled_sequence = model.tokenizer.decode(filled_ids[0], skip_special_tokens=True)
print(f"Filled sequence: {filled_sequence.replace(' ', '')}")
```

### 3. Conditional Generation (e.g., Binders - using `DSM_{ppi}`)
If using `DSM_{ppi}`, the input format is specific for generating a binder (SeqB) for a target (SeqA).

```python
# Example conceptual sketch for DSM_ppi
# from models.modeling_dsm import DSM_ppi # Assuming DSM_ppi is a class or loaded correctly
# model_binder = DSM_ppi.from_pretrained("YourHuggingFaceUser/DSM_650M_ppi_finetuned").to(device).eval()

target_seq = "TARGETSEQUENCEAMINOACIDS"
# For binder generation, the 'interactor' (SeqB) part is what gets generated/filled.
# Start with a fully masked interactor of desired length.
interactor_template_len = 20
interactor_template = model.tokenizer.mask_token * interactor_template_len

combined_input_str = f"{model.tokenizer.cls_token}{target_seq}{model.tokenizer.eos_token}{interactor_template}{model.tokenizer.eos_token}"
# Ensure the total length is within model's max_length

binder_template_tokens = model.tokenizer.encode(combined_input_str, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings).to(device)

generated_binder_ids = model.mask_diffusion_generate( # Assuming 'model' here is an instance of DSM_ppi
    template_tokens=binder_template_tokens,
    steps=int(interactor_template_len / 1) + 10, # More steps for conditional/complex tasks, e.g., one token at a time + refinement
    temperature=1.0,
    remasking="random"
)

generated_full_seq_str = model.tokenizer.decode(generated_binder_ids[0], skip_special_tokens=False)
# Parse out the generated interactor part based on EOS tokens.
# Example: generated_full_seq_str.split(model.tokenizer.eos_token)[1]
print(f"Generated (target+binder): {generated_full_seq_str.replace(' ', '')}")
```

### 4. Autoregressive Generation

```python
# Autoregressive generation
ar_generated_ids = model.auto_regressive_generate(
    batch_size=1,
    length=100,  # Desired length of the protein (excluding CLS/EOS)
    steps=50,    # Number of autoregressive steps
    temperature=0.8,
    remasking="random",
    start_with_methionine=True,
    preview=False
)

ar_generated_sequence = model.tokenizer.decode(ar_generated_ids[0], skip_special_tokens=True)
print(f"Autoregressively generated sequence: {ar_generated_sequence.replace(' ', '')}")
```

**Note:** The exact tokenizer handling and model loading might vary. `model.tokenizer.decode` often adds spaces between tokens; `replace(' ', '')` can remove them for a contiguous sequence. Always check generation parameters like `steps` for optimal results.

## Results

DSM demonstrates strong performance in both protein sequence generation and representation learning, establishing masked diffusion as a powerful paradigm.

-   **Biomimetic Sequence Generation**: Unconditionally generated DSM sequences closely mimic natural protein distributions in terms of amino acid k-mers, predicted secondary structures (JS divergence < 0.01 for AA k-mers), and predicted functional annotations (AV terms, JS divergence ~0.1). This suggests DSM captures underlying biological principles.

-   **Superior Sequence Reconstruction**: DSM models significantly outperform MLM-based ESM2 models in reconstructing sequences from highly corrupted inputs (up to 90% masking).
    -   At 90% masking, DSM achieves an Alignment Score (ASc) of ~0.27, considerably higher than random.
    -   DSM models show higher F1 scores in reconstruction tasks compared to DPLM models, especially at high mask rates.

-   **High-Quality Embeddings**: DSM embeddings match or exceed the quality of those from comparably sized pLMs (ESM2, DPLM) and even larger autoregressive models (ProtCLM 1B) on various downstream tasks evaluated by linear probing. DSM$_{650}$ generally provides the best representations among tested models of similar size.

-   **Effective Binder Design (`DSM_{ppi}`):**
    -   `DSM_{ppi}` fine-tuned on protein-protein interaction data, demonstrates the ability to generate protein binders conditioned on target sequences.
    -   On the BenchBB benchmark, DSM-generated binders (both unconditional DSM and conditional `DSM_{ppi}`) show promising predicted binding affinities, in some cases superior to known binders. For example, designs for EGFR showed high predicted pKd and good structural metrics (ipTM, pTM with AlphaFold3).

-   **Efficiency**: DSM can generate realistic protein sequences from a single forward pass during reconstruction tasks at high mask rates, offering potential efficiency advantages over iterative AR or some discrete diffusion models.

These results highlight DSM's capability to unify high-quality protein representation learning and biologically coherent generative modeling within a single framework.