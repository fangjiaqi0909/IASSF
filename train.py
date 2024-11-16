import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models.IASSF import RestormerUNet, iassf
import setproctitle
from utils import AverageMeter
from datasets.loader_train import PairLoader, TripleLoader
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='IASSF', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=False, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='MSRS', type=str, help='dataset name')
parser.add_argument('--exp', default='outdoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')
parser.add_argument('--proc_name', default='process name', type=str, help='process name')
parser.add_argument('--retrain', action='store_true', help='retrain the model even if a trained model exists')
args = parser.parse_args()

setproctitle.setproctitle(args.proc_name)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
                sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
                add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
                tensor_list.append(add_0)

            batch_list.append(torch.stack(tensor_list, dim=1))

        return torch.cat(batch_list, dim=0)

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()
        self.mse_criterion = torch.nn.MSELoss()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis
        B, C, W, H = image_vis.shape
        image_ir = image_ir.expand(B, C, W, H)
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(generate_img, x_in_max)
        # Gradient
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        B, C, K, W, H = y_grad.shape
        ir_grad = ir_grad.expand(B, C, K, W, H)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.maximum(y_grad, ir_grad)
        loss_grad = F.l1_loss(generate_img_grad, x_grad_joint)

        return loss_in, loss_grad

def train(train_loader, dehaze_network, fusion_network, criterion, fusion_loss, optimizer_dehaze, optimizer_fusion,
          scaler, epoch, setting):
    losses_total = AverageMeter()
    losses_criterion = AverageMeter()
    losses_in = AverageMeter()
    losses_grad = AverageMeter()

    torch.cuda.empty_cache()
    dehaze_network.train()
    fusion_network.train()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{setting['epochs']}", dynamic_ncols=True, leave=True, miniters=5)
    for batch_idx, batch in enumerate(pbar):
        vi_img = batch['vi'].cuda().float()
        ir_img = batch['ir'].cuda().float()
        gt_img = batch['gt'].cuda().float()

        with autocast(enabled=args.no_autocast):
            output_dehaze, output_ir = dehaze_network(vi_img, ir_img)
            output_fusion = fusion_network(output_dehaze, output_ir)

            loss_criterion = 2 * criterion(output_dehaze, gt_img)
            loss_in, loss_grad = fusion_loss(gt_img, ir_img, output_fusion)
            loss_total = loss_in + loss_grad + loss_criterion

        losses_total.update(loss_total.item())
        losses_criterion.update(loss_criterion.item())
        losses_in.update(loss_in.item())
        losses_grad.update(loss_grad.item())

        optimizer_dehaze.zero_grad()
        optimizer_fusion.zero_grad()

        scaler.scale(loss_total).backward()
        scaler.step(optimizer_dehaze)
        scaler.step(optimizer_fusion)
        scaler.update()

        current_lr_dehaze = optimizer_dehaze.param_groups[0]['lr']
        current_lr_fusion = optimizer_fusion.param_groups[0]['lr']

        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'total_loss': f'{losses_total.avg:.4f}',
                'criterion_loss': f'{losses_criterion.avg:.4f}',
                'intensity_loss': f'{losses_in.avg:.4f}',
                'gradient_loss': f'{losses_grad.avg:.4f}',
                'lr_dehaze': f'{current_lr_dehaze:.6f}',
                'lr_fusion': f'{current_lr_fusion:.6f}',
            })
        sys.stdout.flush()

    tqdm.write(f"Epoch {epoch}/{setting['epochs']} - "
               f"Total Loss: {losses_total.avg:.4f}, "
               f"Criterion Loss: {losses_criterion.avg:.4f}, "
               f"Intensity Loss: {losses_in.avg:.4f}, "
               f"Gradient Loss: {losses_grad.avg:.4f}")
    return losses_total.avg


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        print('json file does not exist')
        exit(1)
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    dehaze_network = iassf().cuda()
    fusion_network = RestormerUNet().cuda()
    fusion_loss = Fusionloss()
    criterion = nn.L1Loss()

    optimizer_dehaze = torch.optim.AdamW(dehaze_network.parameters(), lr=setting['lr'])
    optimizer_fusion = torch.optim.AdamW(fusion_network.parameters(), lr=setting['lr'])

    scheduler_dehaze = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_dehaze,
        T_max=setting['epochs'],
        eta_min=setting['lr'] * 1e-2
    )
    scheduler_fusion = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_fusion,
        T_max=setting['epochs'],
        eta_min=setting['lr'] * 1e-2
    )

    scaler = GradScaler()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dataset = TripleLoader(dataset_dir, 'train', 'train',
                                 setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    save_dir = os.path.join(args.save_dir, args.exp, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 0
    checkpoint_path = os.path.join(save_dir, f'{args.model}_checkpoint.pth')
    if os.path.exists(checkpoint_path) and not args.retrain:
        print('==> Loading checkpoint...')
        checkpoint = torch.load(checkpoint_path)
        dehaze_network.load_state_dict(checkpoint['dehaze_network_state_dict'])
        fusion_network.load_state_dict(checkpoint['fusion_network_state_dict'])
        optimizer_dehaze.load_state_dict(checkpoint['optimizer_dehaze_state_dict'])
        optimizer_fusion.load_state_dict(checkpoint['optimizer_fusion_state_dict'])
        scheduler_dehaze.load_state_dict(checkpoint['scheduler_dehaze_state_dict'])
        scheduler_fusion.load_state_dict(checkpoint['scheduler_fusion_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'==> Resuming training from epoch {start_epoch}')

    if not os.path.exists(os.path.join(save_dir, f'{args.model}_dehaze.pth')) or \
            not os.path.exists(os.path.join(save_dir, f'{args.model}_fusion.pth')) or \
            args.retrain:
        print('==> Start training, current model name: ' + args.model)
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

        for epoch in range(start_epoch, setting['epochs'] + 1):
            loss = train(train_loader, dehaze_network, fusion_network, criterion, fusion_loss,
                         optimizer_dehaze, optimizer_fusion, scaler, epoch, setting)
            writer.add_scalar('train_loss', loss, epoch)

            scheduler_dehaze.step()
            scheduler_fusion.step()

            if epoch % 100 == 0 and epoch != 0:
                torch.save({
                    'epoch': epoch,
                    'dehaze_network_state_dict': dehaze_network.state_dict(),
                    'fusion_network_state_dict': fusion_network.state_dict(),
                    'optimizer_dehaze_state_dict': optimizer_dehaze.state_dict(),
                    'optimizer_fusion_state_dict': optimizer_fusion.state_dict(),
                    'scheduler_dehaze_state_dict': scheduler_dehaze.state_dict(),
                    'scheduler_fusion_state_dict': scheduler_fusion.state_dict(),
                    'scaler_state_dict': scaler.state_dict()
                }, os.path.join(save_dir, f'{args.model}_epoch_{epoch}.pth'))

            torch.save({
                'epoch': epoch,
                'dehaze_network_state_dict': dehaze_network.state_dict(),
                'fusion_network_state_dict': fusion_network.state_dict(),
                'optimizer_dehaze_state_dict': optimizer_dehaze.state_dict(),
                'optimizer_fusion_state_dict': optimizer_fusion.state_dict(),
                'scheduler_dehaze_state_dict': scheduler_dehaze.state_dict(),
                'scheduler_fusion_state_dict': scheduler_fusion.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }, checkpoint_path)

        torch.save({
            'dehaze_network_state_dict': dehaze_network.state_dict(),
            'fusion_network_state_dict': fusion_network.state_dict()
        }, os.path.join(save_dir, f'{args.model}_final.pth'))

    else:
        print('==> Existing trained model')
        exit(1)