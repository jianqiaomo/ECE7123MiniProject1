## read model file. Taken from Brightspace
import torch
from project1_model import project1_model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = project1_model().to(device)
model_path = './project1_model.pt'
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)