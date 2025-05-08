# Multi-Target Protein Binder Design

This script generates protein designs for multiple targets and evaluates their binding affinities using ESM-Diff and the Synthyra API.

## Requirements

- Python 3.7+
- PyTorch
- Pandas
- tqdm
- huggingface_hub
- safetensors

## Usage

Run the script with:

```bash
python -m design.multi_target_binder [options]
```

### Command Line Options

- `--token`: HuggingFace token for downloading the model (optional)
- `--num_samples`: Number of designs to generate per target (default: 50)
- `--test`: Use test data instead of calling the API (for debugging)
- `--targets`: Specific targets to design for (e.g., EGFR IL-7Ralpha PD-L1)
- `--batch_size`: Batch size for the model (default: 1)
- `--api_batch_size`: Batch size for the API calls (default: 25)

### Examples

Generate designs for all available targets:
```bash
python -m design.multi_target_binder --num_samples 100
```

Generate designs for specific targets:
```bash
python -m design.multi_target_binder --targets EGFR IL-7Ralpha --num_samples 50
```

## Output Files

The script generates three CSV files:

1. `all_designs.csv`: Contains all generated designs with their predicted binding affinities
2. `design_summary.csv`: Summary statistics for each target including prediction accuracy and improvement metrics
3. `better_designs.csv`: Designs with higher predicted binding affinity than the original template, sorted by target and predicted affinity

## Summary Statistics

The summary includes:

- Target name
- True template pKd (experimentally determined)
- Predicted template pKd
- Prediction error metrics
- Number and percentage of designs with improved binding
- Best design pKd and improvement over template
- Binder source information

## Implementation Details

The script:

1. Loads the ESM-Diff model for protein design
2. Processes each target sequentially 
3. For each target:
   - Generates masked versions of the template sequence
   - Uses ESM-Diff to fill in the masked regions
   - Predicts binding affinity for all designs
4. Compiles results and generates summary statistics

The script uses multi-threading to parallelize the affinity prediction calls, making efficient use of system resources.

## Available Targets

Currently implemented targets with complete information:

- EGFR
- BBF-14
- BHRF1
- IL-7Ralpha
- PD-L1 