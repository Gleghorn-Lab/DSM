import torch


def load_teacher_model(teacher_model_path: str, device: str | torch.device, attn_backend: str = "flex"):
    model_lower = teacher_model_path.lower()
    print(f"Loading Teacher Model from {teacher_model_path}...")

    if "dplm2" in model_lower:
        from models.FastPLMs.dplm2_fastplms.modeling_dplm2 import DPLM2Model

        teacher_model = DPLM2Model.from_pretrained(
            teacher_model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
        ).eval()
    elif "dplm" in model_lower:
        from models.FastPLMs.dplm_fastplms.modeling_dplm import DPLMModel

        teacher_model = DPLMModel.from_pretrained(
            teacher_model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
        ).eval()
    elif ("esm2" in model_lower) or ("fastesm" in model_lower):
        from models.FastPLMs.esm2.modeling_fastesm import FastEsmModel

        teacher_model = FastEsmModel.from_pretrained(
            teacher_model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
        ).eval()
    elif "e1" in model_lower:
        from dsm2.e1 import E1ForMaskedLM

        teacher_model = E1ForMaskedLM.from_pretrained(
            teacher_model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
        ).eval()
    else:
        from models.FastPLMs.esm_plusplus.modeling_esm_plusplus import ESMplusplusModel

        teacher_model = ESMplusplusModel.from_pretrained(
            teacher_model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
        ).eval()

    try:
        teacher_model.attn_backend = attn_backend
    except Exception:
        try:
            teacher_model.attn_backend = "flex"
        except Exception:
            pass
    for param in teacher_model.parameters():
        param.requires_grad = False

    return teacher_model
