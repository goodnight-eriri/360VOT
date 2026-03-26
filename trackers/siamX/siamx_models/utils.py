"""Model utility functions for siamX."""

import torch


def load_pretrain(model: torch.nn.Module, pretrain_path: str, strict: bool = False) -> torch.nn.Module:
    """Load pretrained weights into a model.

    Args:
        model: the PyTorch model to load weights into.
        pretrain_path: path to the ``.pth`` checkpoint file.
        strict: whether to enforce strict key matching.

    Returns:
        model with loaded weights.
    """
    checkpoint = torch.load(pretrain_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Strip leading 'module.' prefix produced by DataParallel/DistributedDataParallel
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Remap keys from external SiamFC checkpoints (e.g. huanglianghua/siamfc-pytorch)
    # where keys use 'features.feature.*' instead of this repo's 'backbone.features.*'
    state_dict = {
        ('backbone.features.' + k[len('features.feature.'):] if k.startswith('features.feature.') else k): v
        for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"[siamX] Missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"[siamX] Unexpected keys in checkpoint: {unexpected}")
    return model
