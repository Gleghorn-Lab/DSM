# Evaluating unconditional generation


First we tune the generation parameters with `unconditional_generation_tuning.py`

```
python -m evaluation.unconditional_generation_tuning --token HF_TOKEN --sweep_temp --step_divisor LARGE_NUMBER_FOR_SPEED
python -m evaluation.unconditional_generation_tuning --token HF_TOKEN --sweep_step --temperature USE_BEST_FROM_ABOVE
```

Then, generate the corpus of sequences, tracking the natural and generated

```
python -m evaluation.unconditional_generation --token HF_TOKEN --output_path CSV_PATH
```

Now, we need to predict the secondary structures and protein annotations of each the natural and generated sequences

Secondary structure

```
python -m evaluation.ss_pred --token HF_TOKEN --input_path CSV_PATH --output_path CSV_PATH_SS
```

Annotations

```
python -m evaluation.annotate_comparisons --token HF_TOKEN --input_path CSV_PATH_SS --output_path CSV_PATH_SS_ANN
```

Now, we can compare the distributions of sequences, structures, and properties

```
python -m evaluation.compare_distributions --input_path CSV_PATH_SS_ANN --output_path DISTRIBUTIONS
```

And we can plot the results

```
python -m evaluation.plot_distribution_comparisons
```