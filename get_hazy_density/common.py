import torch
from torch import nn
from torchvision.io import read_image
from torchvision.transforms import Resize


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

device = get_device()

def image_to_tensor(image_path, device):
    img = read_image(image_path).float()
    img = Resize((256, 256))(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    return img

class ChannelReducer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelReducer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        return self.conv(x)

reducer = ChannelReducer(6, 3).to(device)