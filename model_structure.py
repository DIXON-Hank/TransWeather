from mymodel import UNetTransformerWithAttentionFusion
from torchinfo import summary
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetTransformerWithAttentionFusion()
model = model.to(device)

summary(model, input_size=(1, 3, 256, 512))