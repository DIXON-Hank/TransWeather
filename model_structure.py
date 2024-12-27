from transweather_model import Transweather, UNet
from torchinfo import summary
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transweather()
model = model.to(device)

summary(model, input_size=(1, 3, 1024, 2048))