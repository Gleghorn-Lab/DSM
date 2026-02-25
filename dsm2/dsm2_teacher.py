import torch

from models.FastPLMs.dplm2_fastplms.modeling_dplm2 import DPLM2Model
from models.FastPLMs.dplm_fastplms.modeling_dplm import DPLMModel
from models.FastPLMs.esm2.modeling_fastesm import FastEsmModel
from models.FastPLMs.esm_plusplus.modeling_esm_plusplus import ESMplusplusModel


def load_teacher_model(teacher_model_path: str, device: str | torch.device):
    model_lower = teacher_model_path.lower()
    print(f"Loading Teacher Model from {teacher_model_path}...")

    if "dplm2" in model_lower:
        teacher_model = DPLM2Model.from_pretrained(
            teacher_model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
        ).eval()
    elif "dplm" in model_lower:
        teacher_model = DPLMModel.from_pretrained(
            teacher_model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
        ).eval()
    elif ("esm2" in model_lower) or ("fastesm" in model_lower):
        teacher_model = FastEsmModel.from_pretrained(
            teacher_model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
        ).eval()
    else:
        teacher_model = ESMplusplusModel.from_pretrained(
            teacher_model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
        ).eval()

    teacher_model.attn_backend = "flex"
    for param in teacher_model.parameters():
        param.requires_grad = False

    return teacher_model
