import torch
from mobilevit import MobileViTBlock
import torch
print(torch.__version__)
mb=MobileViTBlock(in_channels=64,transformer_dim=32,ffn_dim=64)
t=torch.rand(size=(1,64,32,32))
f=mb.forward(t)
print(f)
