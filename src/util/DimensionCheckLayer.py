import torch

class DimensionCheckLayer(torch.nn.Module):
    def __init__(self, name, expected_dims):
        super().__init__()
        self.name = name
        self.expected_dims = expected_dims

    def forward(self, x):
        if len(x.shape) != self.expected_dims:
            print(
                f"WARNING: {self.name} has {len(x.shape)} dimensions, expected {self.expected_dims}"
            )
        return x
