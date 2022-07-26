import os
import argparse
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
import numpy as np
from data_util import dataset_factory, collate_factory
from model import model_factory, loss_factory
from model.dlow import dlow_loss_factory
from utils.config import Config
from utils.utils import create_logger
from utils.lr_scheduler import LR_Scheduler
from run_util import trainer_factory
import logging
import random

def prepare_seed(rand_seed):
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

def set_cuda_visible_devices(devs):
    gpus = []
    for dev in devs.split(','):
        dev = dev.strip().lower()
        if dev == 'cpu':
            continue
        if dev.startswith('gpu'):
            dev = dev[3:]
        if '-' in dev:
            l, r = map(int, dev.split('-'))
            gpus.extend(range(l, r + 1))
        else:
            gpus.append(int(dev))

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
    return gpus

def train(num_epoch, trainer, log_dir, logger, model_save_freq=5, save_after_epoch=20):
	loss_best = 1e4
	for i in range(num_epoch):
		logger.info('------------Epoch {0}------------'.format(i))
		avg_loss = trainer.train_epoch()
		if avg_loss<loss_best:
			trainer.save_state(log_dir,i,best=True)
			loss_best = avg_loss
		elif i>save_after_epoch:
			trainer.save_state(log_dir,i)
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg', required=True)
	parser.add_argument('--ckpt_path', type=str, help='finetuning phase')
	parser.add_argument('--log_dir', required=True)
	parser.add_argument('--model_path', type=str, help='Only apply when training dlow,\
	 	load the state_dict per this path before training, overriding cfg')
	parser.add_argument('--save_after_epoch', type=int, default=90)
	parser.add_argument('--debug', action='store_true')

	args = parser.parse_args()

	cfg = Config(args.cfg, tmp=False, create_dirs=True)

	if args.debug:
		cfg.num_workers = 0
		cfg.debug = True
		cfg.batch_size = 2
	if cfg.model_id=='dlow':
		if os.path.isdir(args.model_path):
			model_path = os.path.join(args.model_path, 'model_best.pth')
			if os.path.exists(model_path):
				cfg.model_path = model_path
		elif os.path.exists(args.model_path):
			cfg.model_path = args.model_path

	seed = cfg.seed
	prepare_seed(seed)
	dataset_name = cfg.dataset
	args.log_dir = os.path.join('results',args.log_dir)
	logger = create_logger(args.log_dir)
	log_fd = open(os.path.join(args.log_dir, 'train_dataset_log.txt'),'w')
	train_dataset = dataset_factory[dataset_name](cfg, log_fd, split='train', phase='training')
	log_fd.close()
	log_fd = open(os.path.join(args.log_dir, 'val_dataset_log.txt'),'w')
	val_dataset = dataset_factory[dataset_name](cfg, log_fd, split='val', phase='training')
	log_fd.close()

	dynamic_padding = cfg.get('dynamic_padding', False)
	if dynamic_padding:
		collate_fn = collate_factory['dynamic_padding']
	else:
		collate_fn = None
	train_data_loader = DataLoader(
                        train_dataset, shuffle=True, batch_size=cfg.batch_size,
			num_workers=cfg.num_workers, pin_memory=True, 
			worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
			collate_fn=collate_fn
		)

	device = 'cuda'
	
	model = model_factory[cfg.model_id](cfg)
	print(model)
	if args.ckpt_path:
		print(args.ckpt_path)
		model_path = os.path.join(args.ckpt_path)
		model_dict = {}     
		model_cp = torch.load(model_path)
		for key in model_cp['model'].keys():
			if 'moco' in key or "mask_" in key:
				continue
			model_dict[key] = model_cp['model'][key]
		model.load_state_dict(model_dict, strict=False)
		print('load success')

	if cfg.model_id == 'dlow':
		loss_factory = dlow_loss_factory

	loss_names = cfg.loss_cfg.keys()
	loss_criterions = {n:loss_factory[n](cfg) for n in loss_names}
	model.cuda()

	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
	if 'clip' in cfg.loss_cfg and cfg.loss_cfg['clip']['learn_temperature'] and cfg.loss_cfg['clip'].do:
		optimizer.add_param_group({'params':[loss_criterions['clip'].temperature_clip], 'lr':1e-6})
	if 'moco' in cfg.loss_cfg and cfg.loss_cfg['moco']['learn_temperature'] and cfg.map_moco:
		optimizer.add_param_group({'params':[loss_criterions['moco'].temperature_moco], 'lr':1e-6})

	steps_per_epoch = len(train_data_loader)
	if cfg.lr_scheduler == 'warmup':
		steps_per_udpate = 1
	else:
		steps_per_udpate = len(train_data_loader)
	lr_scheduler = LR_Scheduler(cfg, optimizer, steps_per_udpate, steps_per_epoch)

	logger.info(cfg.yml_dict)

	trainer = trainer_factory[cfg.model_id](cfg, model, device, train_data_loader, \
			loss_criterions, optimizer, lr_scheduler, logger=logger, print_freq=100)

	train(cfg.num_epochs, trainer, args.log_dir, logger, model_save_freq=cfg.model_save_freq,
			save_after_epoch=args.save_after_epoch)




