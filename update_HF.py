from huggingface_hub import HfApi, login


DSM_MODELS = [
    'GleghornLab/DSM_150',
    'GleghornLab/DSM_650',
    'GleghornLab/DSM_150_ppi_lora',
    'GleghornLab/DSM_650_ppi_lora',
    'GleghornLab/DSM_150_ppi_control',
    'Synthyra/DSM_ppi_full',
    'GleghornLab/production_ss4_model',
    'GleghornLab/production_ss9_model',
]


if __name__ == "__main__":
    # py -m update_HF
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    args = parser.parse_args()

    if args.token:
        login(token=args.token)

    api = HfApi()

    # Create temporary README with YAML front matter for HuggingFace
    with open("README.md", "r", encoding="utf-8") as f:
        original_readme = f.read()
    
    yaml_front_matter = """---
library_name: transformers
tags: []
---

"""
    
    hf_readme_content = yaml_front_matter + original_readme
    
    with open("README_HF_temp.md", "w", encoding="utf-8") as f:
        f.write(hf_readme_content)

    for path in DSM_MODELS:
        # Upload license file
        api.upload_file(
            path_or_fileobj="LICENSE",
            path_in_repo="LICENSE",
            repo_id=path,
            repo_type="model",
        )

        # Upload README with YAML front matter
        api.upload_file(
            path_or_fileobj="README_HF_temp.md",
            path_in_repo="README.md",
            repo_id=path,
            repo_type="model",
        )
    
    # Clean up temporary file
    import os
    os.remove("README_HF_temp.md")