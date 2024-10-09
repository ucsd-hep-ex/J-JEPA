import sys
import torch
from src.models.ParT.ParticleTransformerEncoder import ParTEncoder
from src.options import Options


if __name__ == "__main__":
    options = Options()
    options.fc_params = [(128, 0), (256, 0)]
    options.display()
    encoder = ParTEncoder(options=options)

    batch_size = 16
    N_ptcls = 50
    x = v = torch.rand((batch_size, 4, N_ptcls))

    mask_shape = (batch_size, 1, N_ptcls)
    mask = torch.rand(mask_shape) > 0.5
    encoded_features = encoder(x, v, mask)
    print(f"Encoded features shape: {encoded_features.shape}")
