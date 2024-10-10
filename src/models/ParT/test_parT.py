import sys
import torch
from src.models.ParT.ParticleTransformerEncoder import ParTEncoder, ParTPredictor
from src.options import Options


if __name__ == "__main__":
    options = Options()
    options.fc_params = [(128, 0), (256, 0), (512, 0)]
    options.display()

    # test ParT encoder
    encoder = ParTEncoder(options=options)
    print("encoder", encoder)

    batch_size = 16
    N_ctxt = 50
    N_trgt = 30
    x = v = torch.rand((batch_size, 4, N_ctxt))

    ctxt_mask_shape = (batch_size, 1, N_ctxt)
    ctxt_particle_mask = torch.rand(ctxt_mask_shape) > 0.5
    encoded_features = encoder(x, v, ctxt_particle_mask)
    print(f"Encoded features shape: {encoded_features.shape}")

    # test ParTPredictor
    predictor = ParTPredictor(options=options)
    print("predictor", predictor)
    trgt_mask_shape = (batch_size, 1, N_trgt)
    trgt_particle_mask = torch.rand(trgt_mask_shape) > 0.5
    target_particle_features = torch.rand((batch_size, N_trgt, 4))
    context_features = torch.rand((batch_size, N_ctxt, 4))
    output = predictor(
        encoded_features,
        ctxt_particle_mask,
        trgt_particle_mask,
        target_particle_features,
        context_features,
    )
    print(f"Predictor Output shape: {output.shape}")
