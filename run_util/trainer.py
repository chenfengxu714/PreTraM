import sys, os
import torch
from tqdm import tqdm
from .base_runner import BaseTrainer
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.utils import AverageMeter

from PIL import ImageFilter
import random


class AgentFormerTrainer(BaseTrainer):
	def __init__(self, cfg, model, device, data_loader, criterions, optimizer, lr_scheduler, logger=None, print_freq=100):
		super().__init__(model, data_loader, optimizer, lr_scheduler, logger=logger, print_freq=print_freq)
		self.criterions = criterions
		self.loss_keys = criterions.keys()
		self.loss_meters = {k:AverageMeter() for k in self.loss_keys}
		self.optimizer = optimizer
		self.lr_scheduler = lr_scheduler 
		self.print_freq = print_freq
		self.model = model
		self.n_samples = cfg.sample_k
		self.profiling = cfg.get('profiling', False)
		self.total_iter = len(data_loader)
		self.cfg = cfg
		self.cur_batch = None
		self.device = device
		self.debug = cfg.get('debug', False)
		
	def prepare_data(self, data):
		bsz = data['pre_input'].shape[0]
		n_agent = data['pre_input'].shape[2]
		fut_frames = self.cfg.future_frames
		past_frames = self.cfg.past_frames
		#pre_input BS, past_frame, agent num, channel
		data['pre_input'] = data['pre_input'].reshape(bsz, past_frames*n_agent, -1).\
					transpose(0, 1).contiguous().to(self.device)
		data['cur_input'] = data['cur_input'].reshape(bsz, n_agent, -1).transpose(0, 1).\
					contiguous().to(self.device)
		data['future_input'] = data['future_input'].reshape(bsz, fut_frames*n_agent, -1).\
					transpose(0,1).contiguous().to(self.device)
		data['pre_mask'] = data['pre_mask'].transpose(0,2).contiguous().to(self.device)
		data['fut_mask'] = data['fut_mask'].transpose(0,2).contiguous().to(self.device)
		data['cur_mask'] = data['pre_mask'][:,-1,:].to(self.device)
		agent_maps = data['agent_maps']
		width, height = agent_maps.shape[-1], agent_maps.shape[-2]
		data['agent_maps'] = agent_maps.reshape(bsz, n_agent, -1, height, width).to(self.device)
		if self.cfg.map_moco:
			data['agent_maps_moco'] = data['agent_maps_moco'].reshape(bsz, -1, 3, height, width).to(self.device)
			data['agent_maps_moco_augment'] = data['agent_maps_moco'].clone()
			
		data['state'] = data['state'].float().to(self.device)
		data['class'] = data['class'].float().to(self.device)
		
		if 'heading' in self.cfg.dec_input_type:
			data['heading_vec'] = data['heading_vec'].transpose(0,1).contiguous().to(self.device)
			# data['cur_motion'] = data['cur_motion'].reshape(bsz, n_agent, -1).transpose(0,1).contiguous()
		data['fut_motion_orig'] = data['fut_motion_orig'].transpose(0,1).transpose(1,2).contiguous().to(self.device)
		data['scene_orig'] = data['scene_orig'].unsqueeze(0).unsqueeze(1).to(self.device)
		data['pastEn_self_padding_mask'] = data['pastEn_self_padding_mask'].to(self.device)
		data['futEn_self_padding_mask'] = data['futEn_self_padding_mask'].to(self.device)
		data['futEn_cross_padding_mask'] = data['futEn_cross_padding_mask'].to(self.device)
		self.cur_batch = data

	def train_iteration(self, data):
		self.prepare_data(data)
		fut_motion_pred, p_z, q_z, clip_dict = self.model(data)
		if self.cfg.loss_cfg.clip.pretrain:
			return None, None, None, None, clip_dict
		fut_motion_samples, _ = self.model.inference(data, self.n_samples)
		fut_motion_pred, fut_motion_samples = self.postprocessing(fut_motion_pred, fut_motion_samples)
		return fut_motion_pred, fut_motion_samples, p_z, q_z, clip_dict

	def postprocessing(self, fut_motion_pred, fut_motion_samples):
		if self.cfg.pred_type=='orig':
			# in world coordinate
			pass
		elif self.cfg.pred_type=='scene_norm': # convert to world coordinate
			fut_motion_pred += self.cur_batch['scene_orig']
			fut_motion_samples += self.cur_batch['scene_orig']
		elif self.cfg.pred_type=='vel': # TODO add velocity
			raise NotImplementedError
		else:
			raise NotImplementedError
		return fut_motion_pred, fut_motion_samples

	def compute_loss(self, output, data):
		fut_motion_pred, fut_motion_samples, p_z, q_z, output = output
		loss_keys = self.loss_keys
		total_loss = 0
		num_agents=self.cur_batch['agent_num'].to(self.device)
		
		if self.cfg.map_moco:
			orig_map_logist = output['orig_map_logist']
			augments_logist = output['augments_logist']
			if orig_map_logist.dim() == 3:
				orig_map_logist = orig_map_logist.flatten(0, 1)
			if augments_logist.dim() == 3:
				augments_logist = augments_logist.flatten(0, 1)

			moco_loss, unweighted =self.criterions['moco'](orig_map_logist, augments_logist, clip=False)			
			total_loss+=moco_loss
			self.loss_meters['moco'].update(moco_loss.item())
			
		if 'clip' in loss_keys and self.cfg.loss_cfg.clip.do:
			traj_map_logist = output['traj_map_logist'] #bs, dynamic_padding_agentNum, C
			traj_stateclass_logist = output['traj_stateclass_logist']
			
			if traj_map_logist.dim() == 3:
				traj_map_logist = traj_map_logist.flatten(0, 1)
			if traj_stateclass_logist.dim() == 3:
				traj_stateclass_logist = traj_stateclass_logist.flatten(0, 1)
			
			clip_losses = 0
			for name in self.cfg.loss_cfg.clip.contrast_objective:
				key = name + '_logist'
				target_logist = output[key] #bs, max_agentnum, C
				if target_logist.dim() == 3:
					target_logist = target_logist.flatten(0, 1)
					if 'map' in key:
						loss, unweighted =self.criterions['clip'](traj_map_logist, target_logist, data['cur_mask'])
					elif 'state_class' in key:
						loss, unweighted =self.criterions['clip'](traj_stateclass_logist, target_logist, data['cur_mask'])
				clip_losses += loss
			
			total_loss += clip_losses
			self.loss_meters['clip'].update(clip_losses.item())
		
		if self.cfg.loss_cfg.clip.pretrain:
			self.total_loss_meter.update(total_loss.item())
			return total_loss
		
		if 'mse' in loss_keys:
			mse_loss, unweighted = self.criterions['mse'](data['fut_motion_orig'], fut_motion_pred, \
					data['fut_mask'], num_agents=num_agents)
			total_loss += mse_loss
			self.loss_meters['mse'].update(unweighted.item())

		if 'kld' in loss_keys:
			kld_loss, unweighted = self.criterions['kld'](p_z,q_z,data['cur_mask'],num_agents=num_agents)
			total_loss += kld_loss 
			self.loss_meters['kld'].update(unweighted.item())

		if 'sample' in loss_keys:
			sample_loss, unweighted = self.criterions['sample'](data['fut_motion_orig'], fut_motion_samples, \
				data['fut_mask'], num_agents=num_agents)
			total_loss += sample_loss
			self.loss_meters['sample'].update(unweighted.item())
		
		self.total_loss_meter.update(total_loss.item())
		return total_loss

	def generate_msg(self, iter_num):
		profiling_str = ''
		if self.profiling:
			pass
		loss_str = 'total loss:{0:.3f}({1:.3f})\t'.format(self.total_loss_meter.val, self.total_loss_meter.avg) +\
			''.join(['{0} loss:{1:.3f}({2:.3f})\t'.format(name, self.loss_meters[name].val, self.loss_meters[name].avg) for name in self.loss_keys])
		msg = '[{0}/{1}] {2} {3}'.format(iter_num, self.total_iter, profiling_str, loss_str)
		return msg


class DLowTrainer(BaseTrainer):
	# need to specify: train_iteration, compute_loss, generate_msg
	def __init__(self, cfg, model, device, data_loader, criterions, optimizer, lr_scheduler, logger=None, print_freq=100):
		super().__init__(model, data_loader, optimizer, lr_scheduler, logger=logger, print_freq=print_freq)
		self.criterions = criterions
		self.loss_keys = criterions.keys()
		self.loss_meters = {k:AverageMeter() for k in self.loss_keys}
		self.optimizer = optimizer
		self.lr_scheduler = lr_scheduler 
		self.print_freq = print_freq
		self.model = model
		self.pre_cfg = self.model.pre_model.cfg
		# init_model
		state_dict = torch.load(cfg.model_path, map_location='cpu')
		self.model.pre_model.load_state_dict(state_dict['model'], strict=True)
		self.n_samples = cfg.sample_k
		self.profiling = cfg.get('profiling', False)
		self.total_iter = len(data_loader)
		self.cfg = cfg
		self.cur_batch = None
		self.device = device
		self.agentformerTrainer = AgentFormerTrainer(self.pre_cfg, model.pre_model, device, data_loader, \
				criterions, optimizer, lr_scheduler, logger)

	def prepare_data(self, data):
		self.agentformerTrainer.prepare_data(data)
		data['agent_num']=data['agent_num'].to(self.device)
		self.cur_batch = data

	def train_iteration(self, data):
		self.prepare_data(data)
		fut_motion_samples, p_z, sample_z = self.model(data)
		fut_motion_samples, p_z, sample_z = self.postprocessing(fut_motion_samples, p_z, sample_z)
		return fut_motion_samples, p_z, sample_z

	def postprocessing(self, fut_motion_samples, p_z, sample_z):
		if self.pre_cfg.pred_type=='orig':
			# in world coordinate
			pass
		elif self.pre_cfg.pred_type=='scene_norm': # convert to world coordinate
			fut_motion_samples += self.cur_batch['scene_orig']
		elif self.pre_cfg.pred_type=='vel': # TODO add velocity
			raise NotImplementedError
		else:
			raise NotImplementedError
		return fut_motion_samples, p_z, sample_z

	def compute_loss(self, output, data):
		fut_motion_samples, p_z, sample_z = output
		total_loss = 0
		if 'diverse' in self.loss_keys:
			# fut_mask = data['fut_mask'].transpose(0,2).contiguous()
			div_loss, loss_unweighted = self.criterions['diverse'](fut_motion_samples, data['fut_mask'])
			total_loss += div_loss
			self.loss_meters['diverse'].update(div_loss.item())

		if 'kld' in self.loss_keys:
			kld_loss, loss_unweighted = self.criterions['kld'](p_z, sample_z, data['cur_mask'],num_agents=data['agent_num'])
			total_loss += kld_loss
			self.loss_meters['kld'].update(kld_loss.item())

		if 'recon' in self.loss_keys:
			recon_loss, loss_unweighted = self.criterions['recon'](data['fut_motion_orig'], fut_motion_samples, \
				data['fut_mask'], num_agents=data['agent_num'])
			total_loss += recon_loss
			self.loss_meters['recon'].update(recon_loss.item())
		self.total_loss_meter.update(total_loss.item())
		return total_loss

	def generate_msg(self, iter_num):
		profiling_str = ''
		if self.profiling:
			pass
		loss_str = 'total loss:{0:.3f}({1:.3f})\t'.format(self.total_loss_meter.val, self.total_loss_meter.avg) +\
			''.join(['{0} loss:{1:.3f}({2:.3f})\t'.format(name, self.loss_meters[name].val, self.loss_meters[name].avg) for name in self.loss_keys])
		msg = '[{0}/{1}] {2} {3}'.format(iter_num, self.total_iter, profiling_str, loss_str)
		return msg

trainer_factory = {
	'agentformer': AgentFormerTrainer,
	'dlow': DLowTrainer
}
