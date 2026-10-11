"""PT2E APIs moved from torch.ao.quantization to TorchAO."""

try:
    from torchao.quantization.pt2e import allow_exported_model_train_eval
except (ImportError, AttributeError):  # Older Torch/TorchAO versions.
    from torch.ao.quantization import allow_exported_model_train_eval
