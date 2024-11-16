from torch import nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from get_hazy_density.common import device
from get_hazy_density.dcp import Dehaze
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(self, image_paths, device, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(img)
        img = img.to(self.device)
        return img


def get_fog_intensity(img):
    dehaze = Dehaze(img)
    fog_intensity =1 - dehaze.dehaze()
    return fog_intensity



def fog_intensity_to_tensor(vi_feature):
    N, C, H, W = vi_feature.shape

    vi_feature_processed = vi_feature.mean(dim=1, keepdim=True)
    vi_feature_processed = vi_feature_processed.repeat(1, 3, 1, 1)
    feature_np = vi_feature_processed.permute(0, 2, 3, 1).detach().cpu().numpy()

    fog_intensitys = []
    for feature in feature_np:
        feature_normalized = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
        feature_uint8 = (feature_normalized * 255).astype(np.uint8)
        fog_intensity = get_fog_intensity(feature_uint8)
        fog_intensitys.append(fog_intensity)

    fog_intensitys = np.stack(fog_intensitys)
    fog_intensitys_tensor = torch.tensor(fog_intensitys)[:, None, :, :].float().to(vi_feature.device)
    fog_intensitys_tensor = fog_intensitys_tensor.expand(-1, C, -1, -1)

    return fog_intensitys_tensor
