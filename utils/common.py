import numpy as np
import cv2
import torch
from torch import nn
from torchvision.io import read_image
from torchvision.transforms import Resize

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class ListAverageMeter(object):
	def __init__(self):
		self.len = 10000
		self.reset()

	def reset(self):
		self.val = [0] * self.len
		self.avg = [0] * self.len
		self.sum = [0] * self.len
		self.count = 0

	def set_len(self, n):
		self.len = n
		self.reset()

	def update(self, vals, n=1):
		assert len(vals) == self.len, 'length of vals not equal to self.len'
		self.val = vals
		for i in range(self.len):
			self.sum[i] += self.val[i] * n
		self.count += n
		for i in range(self.len):
			self.avg[i] = self.sum[i] / self.count
			

def read_img(filename):
	img = cv2.imread(filename)
	return img[:, :, ::-1].astype('float32') / 255.0


def write_img(filename, img):
	img = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
	cv2.imwrite(filename, img)


def hwc_to_chw(img):
	return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
	return np.transpose(img, axes=[1, 2, 0]).copy()

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



def get_device():
	if torch.cuda.is_available():
		device = 'cuda:0'
	else:
		device = 'cpu'
	return device


device = get_device()

reducer = ChannelReducer(6, 3).to(device)


