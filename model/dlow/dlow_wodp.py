from .. import AgentFormer
from ..common import AgentWisePooling, MLP, Normal, extract_and_aggregate_context
import torch
import torch.nn as  nn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.config import Config

class DLow(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		pre_cfg = Config(cfg.pred_cfg)
		num_dist_params = 2 * cfg.nz
		self.pre_cfg = pre_cfg
		self.nz = cfg.nz
		self.sample_k = cfg.sample_k
		self.num_past_frames = cfg.past_frames
		self.pre_model = AgentFormer(pre_cfg)
		self.d_model = pre_cfg.tf_model_dim
		self.pooling = AgentWisePooling(self.num_past_frames)
		self.out_mlp_dim = cfg.get('qnet_mlp', [512, 256])
		self.sampler_net = MLP(self.d_model, self.out_mlp_dim, 'relu')
		self.sampler_A = nn.Linear(self.out_mlp_dim[-1], self.sample_k*cfg.nz)
		self.sampler_b = nn.Linear(self.out_mlp_dim[-1], self.sample_k*cfg.nz)

		self.dec_input_type = cfg.dec_input_type

	def forward(self, data, return_attn=False):
		max_agents = data['cur_input'].shape[0]
		context_dict = {name:data[name] for name in self.pre_model.context_names}
		context_features = extract_and_aggregate_context(context_dict, 
			self.pre_model.contextEncoders, max_agents)
		context_features = torch.cat(context_features, dim=-1)
		# Past encoder
		# repeat the (agent num, bs, context_dim) into (Txagent num, bs, context_dim)
		past_context_features = context_features.repeat(self.num_past_frames, 1, 1)
		if self.pre_cfg.use_traj_enc:
			past_bef, _ = self.pre_model.past_trj_enc(data['pre_input'], padding_mask=data['pastEn_self_padding_mask'])
		else:
			past_bef = data['pre_input']
		
		past_feature, pre_attn = self.pre_model.past_encoder(past_bef,context_features=past_context_features,\
				padding_mask=data['pastEn_self_padding_mask'], return_attn=return_attn)
		p_z_params = self.pre_model.prior_net(self.pooling(past_feature, data['pre_mask'])) # agent, bs, nz*2
		# repeat on dim=1 because it's more convenient for the trajectory sampler to get samples on the bs dimension
		p_z_params = p_z_params.repeat_interleave(self.sample_k, dim=1) # agent, bs*sample, nz*2
		p_z = Normal(params=p_z_params)

		# trajectory sampler
		pooling_feature = self.pooling(past_feature,data['pre_mask'])
		num_agent = pooling_feature.shape[0]
		z_hidden = self.sampler_net(pooling_feature)
		z_A = self.sampler_A(z_hidden).reshape(num_agent, -1, self.nz) # corresponds to the shape of p_z_params
		z_b = self.sampler_b(z_hidden).reshape(num_agent, -1, self.nz)
		logvar = (z_A**2+1e-8).log()
		sample_z = Normal(mu=z_b, logvar=logvar)

		if 'heading' in self.dec_input_type:
			fut_dec_context_features = torch.cat([context_features, data['heading_vec']], dim=-1)
		# Future Decoder
		# the reshape on z_b here is a bit ugly...but no workaround. It's to align with the shape of z sampled from a Dist obj,
		# which is used in agentformer pretraining
		fut_motion_samples, _ = self.pre_model.future_decoder.sample(data['cur_input'], \
			z_b.reshape(num_agent, -1, self.sample_k, self.nz).transpose(1,2).contiguous(), \
			past_feature, context_features=fut_dec_context_features, num_sample=self.sample_k, \
			self_padding_mask=data['futEn_self_padding_mask'],cross_padding_mask=data['futEn_cross_padding_mask'])
		return fut_motion_samples, p_z, sample_z

