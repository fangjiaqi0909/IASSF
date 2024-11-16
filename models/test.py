import os
import argparse
import torch
from torch.utils.data import DataLoader
from utils import write_img, chw_to_hwc
from datasets.loader_test import PairLoader, TripleLoader
from models.IASSF import RestormerUNet, iassf

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='IASSF', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='MSRS', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
args = parser.parse_args()


def load_model_state(model, state_dict):
    model_state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in model_state_dict and v.size() == model_state_dict[k].size():
            new_state_dict[k] = v
        else:
            print(f"Skipping parameter {k} due to size mismatch or key not found")

    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)


def test(test_loader, dehaze_network, fusion_network, result_dir):
    torch.cuda.empty_cache()

    dehaze_network.eval()
    fusion_network.eval()

    output_folder = os.path.join(result_dir, 'imgs')
    os.makedirs(output_folder, exist_ok=True)

    for idx, batch in enumerate(test_loader):
        vi_img = batch['vi'].cuda().float()
        ir_img = batch['ir'].cuda().float()
        filename = batch['filename'][0]

        with torch.no_grad():
            dehaze_feature, dehaze_ir = dehaze_network(vi_img, ir_img)
            output_fusion = fusion_network(dehaze_feature, dehaze_ir)

            output = output_fusion.clamp_(-1, 1)
            output = output * 0.5 + 0.5

        print(f'Test: [{idx}] Processed')

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(output_folder, filename), out_img)


if __name__ == '__main__':
    dehaze_network = iassf().cuda()
    fusion_network = RestormerUNet().cuda()

    saved_model_dir = 'saved_models/iassf/IASSF.pth'

    if os.path.exists(saved_model_dir):
        print('==> Start testing, current model name: ' + args.model)
        state_dict = torch.load(saved_model_dir)
        load_model_state(dehaze_network, state_dict['dehaze_network_state_dict'])
        load_model_state(fusion_network, state_dict['fusion_network_state_dict'])
    else:
        print('==> No existing trained model!')
        exit(0)

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    test_dataset = TripleLoader(dataset_dir, 'test', mode='test')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)


    result_dir = os.path.join(args.result_dir, args.dataset)
    test(test_loader, dehaze_network, fusion_network, result_dir)