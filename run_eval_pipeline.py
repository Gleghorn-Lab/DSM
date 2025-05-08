#!/usr/bin/env python
import argparse
import subprocess
import os
import sys


def run_command(command, description=None):
    """Run a command and check for errors."""
    if description:
        print(f"\n=== {description} ===")
    print(f"Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run the complete ESM-Diff evaluation pipeline")
    parser.add_argument("--token", required=True, help="Hugging Face token")
    parser.add_argument("--data_dir", default="evaluation_data", help="Directory to store output data")
    parser.add_argument("--temp_sweep_divisor", type=int, default=100, 
                        help="Step divisor for temperature sweep (higher is faster)")
    parser.add_argument("--skip_tuning", action="store_true", help="Skip parameter tuning steps")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Define file paths
    base_csv = os.path.join(args.data_dir, "test_generated_sequences.csv")
    ss_csv = os.path.join(args.data_dir, "test_generated_sequences_ss.csv")
    annotated_csv = os.path.join(args.data_dir, "test_generated_sequences_ss_ann.csv")
    dist_output = os.path.join(args.data_dir, "test_distributions")
    
    # Step 1: Parameter tuning (if not skipped)
    if not args.skip_tuning:
        # Tune temperature
        run_command(
            [
                "python", "-m", "evaluation.unconditional_generation_tuning",
                "--token", args.token,
                "--sweep_temp",
                "--step_divisor", str(args.temp_sweep_divisor)
            ],
            "Temperature parameter tuning"
        )
        
        # Find best temperature from outputs (assuming it's printed to stdout)
        # In a real implementation, you'd need to parse the output to get the best temperature
        best_temp = 0.85  # Default value, replace with actual parsing
        
        # Tune step size with best temperature
        run_command(
            [
                "python", "-m", "evaluation.unconditional_generation_tuning",
                "--token", args.token,
                "--sweep_step",
                "--temperature", str(best_temp)
            ],
            "Step size parameter tuning"
        )
    
    # Step 2: Generate sequences
    run_command(
        [
            "python", "-m", "evaluation.unconditional_generation",
            "--token", args.token,
            "--output_path", base_csv
        ],
        "Generating sequences"
    )
    
    # Step 3: Predict secondary structures
    run_command(
        [
            "python", "-m", "evaluation.ss_pred",
            "--token", args.token,
            "--input_path", base_csv,
            "--output_path", ss_csv
        ],
        "Predicting secondary structures"
    )
    
    # Step 4: Annotate comparisons
    run_command(
        [
            "python", "-m", "evaluation.annotate_comparisons",
            "--token", args.token,
            "--input_path", ss_csv,
            "--output_path", annotated_csv
        ],
        "Annotating sequences with protein properties"
    )
    
    # Step 5: Compare distributions
    run_command(
        [
            "python", "-m", "evaluation.compare_distributions",
            "--input_path", annotated_csv,
            "--output_path", dist_output
        ],
        "Comparing distributions"
    )
    
    # Step 6: Plot results
    run_command(
        [
            "python", "-m", "evaluation.plot_distribution_comparisons"
        ],
        "Plotting distribution comparisons"
    )
    
    print("\n=== Evaluation pipeline completed successfully ===")
    print(f"Results stored in: {args.data_dir}")


if __name__ == "__main__":
    main() 