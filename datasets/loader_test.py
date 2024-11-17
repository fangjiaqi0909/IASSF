import os
import random
import numpy as np
import cv2
from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img


def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
    _, C, H, W = imgs[0].shape
    Hc, Wc = [size, size]

    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H - Hc)

    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W - Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][:, :, Hs:(Hs + Hc), Ws:(Ws + Wc)]

    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.ascontiguousarray(np.flip(imgs[i], axis=3))

    if not only_h_flip:
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.ascontiguousarray(np.rot90(imgs[i], rot_deg, (2, 3)))

    return imgs


def align(imgs=[], size=256):
    if size is None:
        return imgs

    _, C, H, W = imgs[0].shape
    Hc, Wc = [size, size]

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][:, :, Hs:(Hs + Hc), Ws:(Ws + Wc)]

    return imgs


class PairLoader(Dataset):
    def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip

        self.root_dir = os.path.join(data_dir, sub_dir)
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'hazy')))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_name = self.img_names[idx]

        vi_path = os.path.join(self.root_dir, 'hazy', img_name)
        ir_path = os.path.join(self.root_dir, 'IR', img_name)

        if not os.path.exists(vi_path) or not os.path.exists(ir_path):
            raise FileNotFoundError(f"One of the image files not found: {vi_path}, {ir_path}")

        vi_img = read_img(vi_path, self.size) * 2 - 1
        ir_img = read_img(ir_path, self.size) * 2 - 1

        if self.mode == 'train':
            [vi_img, ir_img] = augment([vi_img, ir_img], self.size, self.edge_decay, self.only_h_flip)

        if self.mode == 'valid':
            [vi_img, ir_img] = align([vi_img, ir_img], self.size)

        vi_img = vi_img.squeeze(0)
        ir_img = ir_img.squeeze(0)

        return {'vi': vi_img.astype(np.float32), 'ir': ir_img.astype(np.float32), 'filename': img_name}


class TripleLoader(Dataset):
    def __init__(self, data_dir, sub_dir, mode, size=None, edge_decay=0, only_h_flip=False):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.size = size if mode == 'train' else None
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip

        self.root_dir = os.path.join(data_dir, sub_dir)
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_name = self.img_names[idx]

        vi_path = os.path.join(self.root_dir, 'hazy', img_name)
        ir_path = os.path.join(self.root_dir, 'IR', img_name)
        gt_path = os.path.join(self.root_dir, 'GT', img_name)

        if not os.path.exists(vi_path):
            print(f"Vi image file not found: {vi_path}")
            return None
        if not os.path.exists(ir_path):
            print(f"IR image file not found: {ir_path}")
            return None
        if not os.path.exists(gt_path):
            print(f"GT image file not found: {gt_path}")
            return None

        vi_img = read_img(vi_path, self.size) * 2 - 1
        ir_img = read_img(ir_path, self.size) * 2 - 1
        gt_img = read_img(gt_path, self.size) * 2 - 1

        if self.mode == 'train':
            [vi_img, ir_img, gt_img] = augment([vi_img, ir_img, gt_img], self.size, self.edge_decay, self.only_h_flip)

        if self.mode == 'valid':
            [vi_img, ir_img, gt_img] = align([vi_img, ir_img, gt_img], self.size)

        vi_img = vi_img.squeeze(0)
        ir_img = ir_img.squeeze(0)
        gt_img = gt_img.squeeze(0)

        return {
            'vi': vi_img.astype(np.float32),
            'ir': ir_img.astype(np.float32),
            'gt': gt_img.astype(np.float32),
            'filename': img_name
        }


def hwc_to_chw(image):
    return np.transpose(image, (2, 0, 1))


def read_img(filepath, size=None):
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Image file not found: {filepath}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    img = np.expand_dims(img, axis=0)
    img = np.transpose(img, (0, 3, 1, 2))
    return img
