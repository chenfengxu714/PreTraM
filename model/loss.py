from turtle import forward
import torch
import torch.nn as nn
import numpy as np

class mploss(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.mse_loss = nn.MSELoss()
		self.weight = cfg.loss_cfg['mask_trj'].get('weight', 1)
	
	def forward(self, src, tgt):
		loss_unweighted = self.mse_loss(src, tgt)
		loss = loss_unweighted * self.weight
		return loss, loss_unweighted


class contrastLoss(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.crossEntropy = nn.CrossEntropyLoss(reduction='none')
		self.weight_clip = cfg.loss_cfg['clip']['weight']
		self.weight_moco = cfg.loss_cfg['moco']['weight']
		temp = cfg.loss_cfg['clip']['temperature']
		self.temperature_clip = nn.Parameter(torch.ones([])*np.log(1/temp))
		
		temp = cfg.loss_cfg['moco']['temperature']
		self.temperature_moco = nn.Parameter(torch.ones([])*np.log(1/temp))
		
		self.globalweight = cfg.loss_cfg['clip'].get('globalweight', 1)

	def forward(self, src, tgt, mask=None, clip=True):
		'''
		match traj sequence with image sequence
		'''
		# src, tgt shape: bs*L, ch
		#chenfeng: waiting for add the global CLIP loss
			
		L = src.shape[0]
		if clip:
			sim = torch.mm(src, tgt.transpose(0,1).contiguous()
				)*torch.exp(self.temperature_clip)
		else:
			sim = torch.mm(src, tgt.transpose(0,1).contiguous()
				)*torch.exp(self.temperature_moco)

		if mask is not None:
			mask = mask.transpose(0, 1).flatten(0, 1).unsqueeze(-1)
			mask_booled = mask.mm(mask.T)
			mask_booled = (mask_booled == 0)
			sim.masked_fill_(mask_booled, -1e5)
		
		label = torch.arange(L, dtype=torch.long, device=sim.device)
		loss = (self.crossEntropy(sim, label) \
			+ self.crossEntropy(sim.transpose(0,1).contiguous(), label)) / 2
		if mask is not None:
			loss = torch.sum(loss * mask.squeeze()) / torch.sum(mask)
		else:
			loss = loss.mean()
		return loss*self.weight_clip if clip else loss*self.weight_moco, loss

class Motion_mse(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.normalize = cfg.get('normalize', True)
		self.weight = cfg.loss_cfg['mse']['weight']

	def forward(self, future_motion_gt, pred_motion, mask, **kwargs):
		# print(kwargs['num_agents'], mask)
		# motion: [agent, time, batch, 2], mask: [agent, time, batch]
		diff = future_motion_gt - pred_motion
		diff *= mask.unsqueeze(-1)
		# print(future_motion_gt.shape, pred_motion.shape, mask.shape)
		loss_unweighted = diff.pow(2).sum(dim=(0,1,3)) 
		if self.normalize:
			loss_unweighted /= kwargs['num_agents']
			loss_unweighted = loss_unweighted.mean()
		else:
			loss_unweighted = loss_unweighted.sum()
		loss = loss_unweighted * self.weight
		return loss, loss_unweighted

class KL_divergence(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.normalize = cfg.get('normalize', True)
		# set this to 0 if one does not want it
		self.min_clip = cfg.loss_cfg['kld']['min_clip']
		self.weight = cfg.loss_cfg['kld']['weight']

	def forward(self, p_z_distr, q_z_distr, mask, **kwargs):
		loss_unweighted = q_z_distr.kl(p_z_distr).sum(dim=-1)
		loss_unweighted = (loss_unweighted * mask).sum(dim=0)
		if self.normalize:
			loss_unweighted /= kwargs['num_agents'] # numagent by batch
			loss_unweighted = loss_unweighted.mean()
		else:
			loss_unweighted = loss_unweighted.sum()
		loss_unweighted = loss_unweighted.clamp_min_(self.min_clip)
		loss = loss_unweighted * self.weight
		return loss, loss_unweighted


class Sample_motion_mse(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.normalize = cfg.get('normalize', True)
		self.weight = cfg.loss_cfg['sample']['weight']

	def forward(self, future_motion_gt, sample_motion, mask, **kwargs):
		# [agent sample time batch feature]
		diff = sample_motion - future_motion_gt.unsqueeze(1)
		diff *= mask.unsqueeze(1).unsqueeze(-1)
		dist = diff.pow(2).sum(dim=(-1,-3))
		loss_unweighted = dist.min(dim=1)[0]
		if self.normalize:
			loss_unweighted = loss_unweighted.sum(dim=0) / kwargs['num_agents']
			loss_unweighted = loss_unweighted.mean() # average over agent and batch)
		else:
			loss_unweighted = loss_unweighted.sum()
		loss = loss_unweighted * self.weight
		return loss, loss_unweighted


loss_factory = {
	'mse': Motion_mse,
	'kld': KL_divergence,
	'sample': Sample_motion_mse,
	'clip': contrastLoss,
	'moco': contrastLoss,
	'mask_trj': mploss
}