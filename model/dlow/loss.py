import torch
import torch.nn as nn 
import torch.nn.functional as F

class KL_divergence(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.normalize = cfg.get('normalize', True)
		# set this to 0 if one do not want it
		self.min_clip = cfg.loss_cfg['kld']['min_clip']
		self.weight = cfg.loss_cfg['kld']['weight']

	def forward(self, p_z_distr, q_z_distr, mask, **kwargs):
		# mask: (nagent, bs), binary
		# nagent, sample*bs, nz
		nagent, bs = mask.shape[0],mask.shape[1]
		kld = q_z_distr.kl(p_z_distr).sum(dim=-1) # sum over nz
		kld = kld.reshape(nagent, -1, bs).sum(dim=1) # sum over samples
		# nagent, bs
		loss_unweighted = (kld * mask).sum(dim=0) # sum over nagents
		# bs
		if self.normalize:
			loss_unweighted /= kwargs['num_agents'] # divide by numagent of each instance in batch
			loss_unweighted = loss_unweighted.mean() # mean over batch
		else:
			loss_unweighted = loss_unweighted.sum()
		loss_unweighted = loss_unweighted.clamp_min_(self.min_clip)
		loss = loss_unweighted * self.weight
		return loss, loss_unweighted

class Diversity_loss(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.normalize = cfg.get('normalize', True)
		self.d_scale = cfg.loss_cfg['diverse']['d_scale']
		self.weight = cfg.loss_cfg['diverse']['weight']

	def forward(self, sample_motions, mask, **kwargs): 
		# mask: (agent, fut_frames, bs), binary NTB
		loss_unweighted = 0
		# 
		agent, sample, frames, bsz = sample_motions.shape[0], sample_motions.shape[1], \
				sample_motions.shape[2], sample_motions.shape[3]
		# agent, sample, frames, batchsize, forecast_dim -> bs, agent, sample, frames, dim -> bs*agent, sample, frames*dim
		sample_motions = (sample_motions * mask.unsqueeze(1).unsqueeze(-1)).\
				permute(3,0,1,2,4).contiguous().reshape(bsz*agent, sample, -1) # TODO: test, 
		for agent_motion in sample_motions: 
			dist = F.pdist(agent_motion, 2)**2
			loss_unweighted += (-dist / self.d_scale).exp().mean()
		if self.normalize:
			loss_unweighted /= agent*bsz
		loss = loss_unweighted * self.weight
		return loss, loss_unweighted

class Reconstruction_loss(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.normalize = cfg.get('normalize', True)
		self.weight = cfg.loss_cfg['recon']['weight']

	def forward(self, future_motion_gt, sample_motions, mask, **kwargs):
		# agent, sample, frames, batchsize, forecast_dim
		diff = sample_motions - future_motion_gt.unsqueeze(1)
		diff *= mask.unsqueeze(1).unsqueeze(-1)
		dist = diff.pow(2).sum(dim=(-1,-3))
		loss_unweighted = dist.min(dim=1)[0]
		if self.normalize:
			loss_unweighted = loss_unweighted.sum(dim=0) / kwargs['num_agents'] # average over agent
			loss_unweighted = loss_unweighted.mean() # average over batch
		else:
			loss_unweighted = loss_unweighted.sum()
		loss = loss_unweighted * self.weight
		return loss, loss_unweighted


dlow_loss_factory = {
	'diverse': Diversity_loss,
	'kld': KL_divergence,
	'recon': Reconstruction_loss
}
