import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.jjepa import JJEPA
from src.options import Options
import torch
def load_model(model_path=None, device="cpu"):
    options = Options("src/test_options.json")
    options.num_particles = 30
    model = JJEPA(options).to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    print(model)
    return model


if __name__ == "__main__":
    model = load_model("/mnt/d/physic/data/best_model.pth", device="cpu")


