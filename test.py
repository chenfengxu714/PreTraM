import numpy as np
import argparse
import os
import re
import sys
import subprocess
import shutil

sys.path.append(os.getcwd())
from data_util import dataset_factory, collate_factory
from utils.torch import *
from utils.config import Config
from utils.utils import prepare_seed, print_log, mkdir_if_missing

import torch
from torch.utils.data import DataLoader

from data_util import dataset_factory
from model import model_factory
from utils.config import Config
from utils.utils import create_logger, create_summary_writer
from run_util import evaler_factory
from evaluation import evaluate

log = open('results/log_test.txt', 'w')

def test_with_checkpoint(model_file, cfg, dataloader, device, save_dir, logger, writer=None, epoch=None, split='test'):
    logger.info('Evaluating with {}'.format(model_file))
    prepare_seed(cfg.seed)

    model_id = cfg.get('model_id', 'agentformer')
    model = model_factory[model_id](cfg)
    model.to(device)
    model.eval()
    model_dict = {}     
    model_cp = torch.load(model_file)
    for key in model_cp['model'].keys():
        if 'traj_logist_mlp' in key and "map_mlp" in key: #and ??
            logger.info('drop projection')
            continue
        model_dict[key] = model_cp['model'][key]
    model.load_state_dict(model_dict, strict=False)
    logger.info('load success')

    eval_dir = f'{save_dir}/samples'
    evaler = evaler_factory[cfg.model_id](cfg, model, device, dataloader, save_dir, split, log)
    evaler.test_model()

    ade, fde = evaluate(cfg.dataset, eval_dir, split, logger)
    if writer is not None:
        writer.add_scalar('ADE', ade, epoch)
        writer.add_scalar('FDE', fde, epoch)

def test_with_avg_checkpoints(model_files, cfg, dataloader, device, save_dir, logger, writer=None, epoch=None, split='test'):
    logger.info('Evaluating with {}'.format(model_files))
    prepare_seed(cfg.seed)

    model_id = cfg.get('model_id', 'agentformer')
    model = model_factory[model_id](cfg)
    model.to(device)
    model.eval()
    model_dict = {}  
    for model_file in model_files:   
        model_cp = torch.load(model_file)
        for key in model_cp['model'].keys():
            if 'traj_logist_mlp' in key and "map_mlp" in key: #and ??
                logger.info('drop projection')
                continue
            if key in model_dict:
                model_dict[key] += model_cp['model'][key]
            else:
                model_dict[key] = model_cp['model'][key]
    for key in model_dict:
        model_dict[key] /= len(model_files)
    model.load_state_dict(model_dict, strict=False)
    logger.info('load success')

    eval_dir = f'{save_dir}/samples'
    evaler = evaler_factory[cfg.model_id](cfg, model, device, dataloader, save_dir, split, log)
    evaler.test_model()

    ade, fde = evaluate(cfg.dataset, eval_dir, split, logger)
    if writer is not None:
        writer.add_scalar('ADE', ade, epoch)
        writer.add_scalar('FDE', fde, epoch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_eval', default='test')
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--evaluate_freq', type=int, default=1)
    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg)
    # test_log_dir = os.path.join(args.ckpt_dir, 'test_logs'); mkdir_if_missing(test_log_dir)
    logger = create_logger(args.ckpt_dir, 'test')
    writer = create_summary_writer(args.ckpt_dir, 'test')
    # if args.epochs is None:
    #     epochs = [cfg.get_last_epoch()]
    # else:
    #     epochs = [int(x) for x in args.epochs.split(',')]

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and \
                        torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    torch.set_grad_enabled(False)
    
    data_splits = [args.data_eval]
    dataset_name = cfg.dataset
    dynamic_padding = cfg.get('dynamic_padding', False)
    if dynamic_padding:
        collate_fn = collate_factory['dynamic_padding']
    else:
        collate_fn = None
    
    val_dataset = dataset_factory[dataset_name](cfg, log, split=args.data_eval, phase='testing')
            
    val_data_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=1,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn)

    model_files = []
    for i in range(args.max_epoch):
        model_file = 'model_{}.pth'.format(i)
        if os.path.exists(os.path.join(args.ckpt_dir, model_file)):
            model_files.append(os.path.join(args.ckpt_dir, model_file))

    save_dir = f'{args.ckpt_dir}/{args.data_eval}_results'; mkdir_if_missing(save_dir)
    
    model_file = os.path.join(args.ckpt_dir, 'model_best.pth')
    test_with_checkpoint(model_file, cfg, val_data_loader, device, save_dir,\
                     logger, writer=None, split=args.data_eval)

    last_epoch = args.max_epoch+args.evaluate_freq+1
    for i in range(args.max_epoch,-1,-1):
        model_file = 'model_{}.pth'.format(i)
        full_path = os.path.join(args.ckpt_dir, model_file)
        if not os.path.exists(full_path):
            continue
        if last_epoch-i < args.evaluate_freq and i<=0.9*args.max_epoch:
            continue
        last_epoch = i
        test_with_checkpoint(full_path, cfg, val_data_loader, device, save_dir,\
                     logger, writer=writer, epoch=i, split=args.data_eval)