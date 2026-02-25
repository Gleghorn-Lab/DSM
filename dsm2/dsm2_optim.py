import torch
import torch.distributed as dist


class MuonAdamWWrapper(torch.optim.Optimizer):
    def __init__(self, muonclip, adamw: torch.optim.AdamW):
        self.muonclip = muonclip
        self.adamw = adamw
        wrapper_param_groups = self.muonclip.param_groups + self.adamw.param_groups
        super().__init__(wrapper_param_groups, defaults={})
        self._muon_group_count = len(self.muonclip.param_groups)
        self.last_s_max: list[list[torch.Tensor]] | None = None
        self._sync_wrapper_state()

    def _sync_wrapper_state(self):
        self.param_groups = self.muonclip.param_groups + self.adamw.param_groups
        merged_state = {}
        merged_state.update(self.muonclip.state)
        merged_state.update(self.adamw.state)
        self.state = merged_state

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        assert self.last_s_max is not None, "MuonClip step requires per-head s_max values from the model forward pass."
        self.muonclip.step(self.last_s_max)
        self.adamw.step()
        self._sync_wrapper_state()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.muonclip.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)
        self.last_s_max = None

    def remove_params(self, params_to_remove):
        remove_param_ids = {id(param) for param in params_to_remove}

        def _prune_param_groups(param_groups):
            updated_param_groups = []
            for group in param_groups:
                group_params = [param for param in group["params"] if id(param) not in remove_param_ids]
                updated_group = dict(group)
                updated_group["params"] = group_params
                updated_param_groups.append(updated_group)
            return updated_param_groups

        self.muonclip.param_groups = _prune_param_groups(self.muonclip.param_groups)
        self.adamw.param_groups = _prune_param_groups(self.adamw.param_groups)
        self._muon_group_count = len(self.muonclip.param_groups)

        for optimizer_state in (self.muonclip.state, self.adamw.state, self.state):
            stale_params = []
            for param in optimizer_state:
                if id(param) in remove_param_ids:
                    stale_params.append(param)
            for param in stale_params:
                del optimizer_state[param]

        self._sync_wrapper_state()

    def state_dict(self):
        self._sync_wrapper_state()
        state_dict = super().state_dict()
        state_dict["last_s_max"] = self.last_s_max
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict_copy = dict(state_dict)
        self.last_s_max = state_dict_copy.pop("last_s_max", None)
        super().load_state_dict(state_dict_copy)
        self.muonclip.param_groups = self.param_groups[:self._muon_group_count]
        self.adamw.param_groups = self.param_groups[self._muon_group_count:]
        self.muonclip.state = self.state
        self.adamw.state = self.state


def partition_dsm2_parameters(model):
    muon_params = []
    adamw_params = []
    attention_params = []

    for name, param in model.named_parameters():
        if (param.ndim >= 2) and ("embed" not in name) and ("lm_head" not in name):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    for block in model.transformer.blocks:
        attention_params.append(block.attn)

    assert len(muon_params) > 0, "No parameters selected for Muon."
    assert len(adamw_params) > 0, "No parameters selected for AdamW."
    assert len(attention_params) > 0, "No attention modules were found for QKClip."
    return muon_params, adamw_params, attention_params


def create_muonclip_optimizer(model, muon_params, attention_params, muon_lr: float, muon_tau: float):
    from models.muonclip.muonclip import MuonClip

    is_ddp = dist.is_initialized()
    rank = dist.get_rank() if is_ddp else 0
    world_size = dist.get_world_size() if is_ddp else 1

    return MuonClip(
        params=muon_params,
        attention_params=attention_params,
        mode="mha",
        metadata={"w_q": "W_q", "w_k": "W_k"},
        n_head=model.config.num_attention_heads,
        tau=muon_tau,
        lr=muon_lr,
        rank=rank,
        world_size=world_size,
    )
