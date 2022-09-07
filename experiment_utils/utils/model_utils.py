import torch
from omegaconf import DictConfig


def load_model_state(model: torch.nn.Module, cfg: DictConfig) -> None:
    model_state = model.state_dict()
    loaded_state = torch.load(f"{cfg.model_load.model_path}{cfg.model_load.file_name}.pt")
    loaded_state_keys = loaded_state.keys()
    model_state_keys = model_state.keys()
    new_keys = [key for key in model_state_keys if key not in loaded_state_keys]
    if new_keys:
        print(f"keys missed in saved weights: {new_keys}")
        for key in new_keys:
            loaded_state[key] = model_state[key]
    missed_keys = [key for key in loaded_state_keys if key not in model_state_keys]
    if missed_keys:
        print(f"New keys in model: {missed_keys}")
        for key in missed_keys:
            if ".bn." in key:
                loaded_state.pop(key)
    wrong_shape = []
    for key in loaded_state.keys():
        if loaded_state[key].shape != model_state[key].shape:
            wrong_shape.append(key)
            loaded_state[key] = loaded_state[key].view_as(model_state[key])
    if wrong_shape:
        print(f"changed {len(wrong_shape)} modules.")
    if cfg.model_load.se_name:
        se_state = torch.load(f"{cfg.model_load.model_path}{cfg.model_load.se_name}.pt")
        for key in se_state.keys():
            loaded_state[key] = se_state[key].view_as(model_state[key])
        print(f"loaded {len(se_state)} se weights {cfg.model_load.se_name}")
    model.load_state_dict(loaded_state)
