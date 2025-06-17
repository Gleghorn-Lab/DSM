from huggingface_hub import HfApi, login


DSM_MODELS = [
    'GleghornLab/DSM_150',
    'GleghornLab/DSM_650',
    'GleghornLab/DSM_150_ppi_lora',
    'GleghornLab/DSM_650_ppi_lora',
    'GleghornLab/DSM_150_ppi_control',
    'Synthyra/DSM_ppi_full',
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

    for path in DSM_MODELS:
        # Upload license file
        api.upload_file(
            path_or_fileobj="LICENSE",
            path_in_repo="LICENSE",
            repo_id=path,
            repo_type="model",
        )

        # Upload README
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=path,
            repo_type="model",
        )