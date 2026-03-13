import torch
import torch.nn as nn
import torchvision.models as models

MODEL_PATH = "plant_disease_model.pth"


def load_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, 3)

    state_dict = torch.load(MODEL_PATH, map_location="cpu")

    if "module." in list(state_dict.keys())[0]:
        new_state = {}
        for k, v in state_dict.items():
            new_state[k.replace("module.", "")] = v
        state_dict = new_state

    model.load_state_dict(state_dict)

    model.eval()

    return model