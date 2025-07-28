import torch

model_path = "D:/Auto_aiops/checkpoints/best_effnet_b3_multilabel.pth"
state_dict = torch.load(model_path, map_location="cpu")

print("Loaded object type:", type(state_dict))

if isinstance(state_dict, dict):
    print("Top-level keys:", state_dict.keys())



