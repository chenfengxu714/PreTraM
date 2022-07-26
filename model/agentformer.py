import torch
import torch.nn as nn
import numpy as np 
from .ContextModel import context_model_factory
from .agentformer_lib import AgentFormerDecoder, AgentFormerEncoder
from .common import Normal, MLP, AgentWisePooling, extract_and_aggregate_context

""" Positional Encoding """
class PositionalAgentEncoding(nn.Module):

	def __init__(self, cfg, traj_enc=False):
		super(PositionalAgentEncoding, self).__init__()
		self.d_model = cfg.get('tf_model_dim') \
			if not traj_enc else cfg.past_trj_encoder.get('tf_model_dim')
		dropout = cfg.tf_dropout
		max_t_len = cfg.get('max_t_len', 200)
		self.dropout = nn.Dropout(p=dropout)
		# max_a_len = cfg.get('max_a_len', 30)
		pe = self.build_pos_enc(max_t_len)
		self.register_buffer('pe', pe)
		# ae = nn.Parameter(torch.randn(max_a_len, 1, self.d_model)*0.1)
		# self.register_buffer('ae', ae)

	def build_pos_enc(self, max_len):
		pe = torch.zeros(max_len, self.d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		return pe
	
	def build_agent_enc(self, max_len):
		ae = torch.zeros(max_len, self.d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
		ae[:, 0::2] = torch.sin(position * div_term)
		ae[:, 1::2] = torch.cos(position * div_term)
		ae = ae.unsqueeze(0).transpose(0, 1)
		return ae

	def get_pos_enc(self, num_t, num_a, t_offset):
		pe = self.pe[t_offset: num_t + t_offset, :]
		pe = pe.repeat_interleave(num_a, dim=0)
		return pe
 
	def get_agent_enc(self, num_t, num_a, a_offset=0):
		ae = self.ae[a_offset: num_a + a_offset, :]
		ae = ae.repeat(num_t, 1, 1)
		return ae

	def forward(self, x, num_a, agent_enc_shuffle=None, t_offset=0, a_offset=0):
		num_t = x.shape[0] // num_a
		pos_enc = self.get_pos_enc(num_t, num_a, t_offset)
		# agent_enc = self.get_agent_enc(num_t, num_a)
		# print(pos_enc.mean(), t_offset)
		if x.ndim == pos_enc.ndim +1:
			pos_enc = pos_enc.unsqueeze(1)
		x += pos_enc
		# x += agent_enc
		return self.dropout(x)

class Past_trj_enc(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.nlayer = cfg.past_trj_encoder.get('nlayer',2)
		self.past_frames = cfg.past_frames
		self.input_dim = 2 * len(cfg.input_type)
		self.d_model = cfg.past_trj_encoder.get('tf_model_dim')
		self.input_fc = nn.Linear(self.input_dim, self.d_model)
		self.AgentformerEncoder = AgentFormerEncoder(cfg, self.nlayer)
		self.time_encoder = PositionalAgentEncoding(cfg, traj_enc=True)
		self.max_agents = None
		self.use_agent_aware = cfg.get('use_agent_aware', True)

	def forward(self, past_motion, padding_mask=None, agent_aware_mask=None, return_attn=False):
		self.max_agents = past_motion.shape[0]//self.past_frames
		num_agents = self.max_agents if self.use_agent_aware else 1
		tf_in = self.input_fc(past_motion)
		tf_in_timed = self.time_encoder(tf_in, num_a=self.max_agents)
		C = tf_in_timed.shape[-1]
		tf_in_timed = tf_in_timed.view(
			self.past_frames, self.max_agents, -1, C)
		tf_in_timed = tf_in_timed.view(self.past_frames, -1, C)
		pasts, _ = self.AgentformerEncoder(tf_in_timed)
		return pasts.view(self.past_frames, self.max_agents, -1, C
			).view(self.past_frames*self.max_agents, -1, C), _

class Future_trj_enc(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.nlayer = cfg.future_trj_encoder.get('nlayer',2)
		self.future_frames = cfg.future_frames
		self.past_frames = cfg.past_frames
		self.input_dim = 2 * len(cfg.input_type)
		self.d_model = cfg.future_trj_encoder.get('tf_model_dim')
		self.input_fc = nn.Linear(self.input_dim, self.d_model)
		self.AgentformerEncoder = AgentFormerEncoder(cfg, self.nlayer)
		self.time_encoder = PositionalAgentEncoding(cfg, traj_enc=True)
		self.max_agents = None
		self.use_agent_aware = cfg.get('use_agent_aware', True)

	def forward(self, future_motion, padding_mask=None, agent_aware_mask=None, return_attn=False):
		self.max_agents = future_motion.shape[0]//self.future_frames
		num_agents = self.max_agents if self.use_agent_aware else 1
		tf_in = self.input_fc(future_motion)
		tf_in_timed = self.time_encoder(tf_in, num_a=self.max_agents, t_offset=self.past_frames)
		C = tf_in_timed.shape[-1]
		tf_in_timed = tf_in_timed.view(
			self.future_frames, self.max_agents, -1, C)
		tf_in_timed = tf_in_timed.view(self.future_frames, -1, C)
		futures, _ = self.AgentformerEncoder(tf_in_timed)
		return futures.view(self.future_frames, self.max_agents, -1, C
			).view(self.future_frames*self.max_agents, -1, C), _

		# return self.AgentformerEncoder(tf_in_timed, mask=padding_mask, agent_aware_mask=agent_aware_mask,\
		# 		 need_weights=return_attn, num_agent=num_agents)

class PastEncoder(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.nlayer = cfg.context_encoder.get('nlayer',6)
		self.past_frames = cfg.past_frames
		self.context_names = cfg.get('context_names',[]) # alter this to add more context
		self.input_dim = 2 * len(cfg.input_type) 
		for name in self.context_names:
			cfg_context = cfg.get('{}_encoder'.format(name))
			self.input_dim += cfg_context.out_dim
		self.d_model = cfg.get('tf_model_dim')
		self.input_fc = nn.Linear(self.input_dim, self.d_model)
		self.AgentformerEncoder = AgentFormerEncoder(cfg, self.nlayer)
		self.time_encoder = PositionalAgentEncoding(cfg)
		self.max_agents = None
		self.use_agent_aware = cfg.get('use_agent_aware', True)

	def forward(self, past_motion, context_features=None, padding_mask=None, agent_aware_mask=None, return_attn=False):
		self.max_agents = past_motion.shape[0]//self.past_frames
		num_agents = self.max_agents if self.use_agent_aware else 1
		if context_features is not None:
			data_in = torch.cat([past_motion, context_features], dim=-1)
			tf_in = self.input_fc(data_in)
		else:
			tf_in = past_motion
		tf_in_timed = self.time_encoder(tf_in, num_a=self.max_agents)
		return self.AgentformerEncoder(tf_in_timed, mask=padding_mask, agent_aware_mask=agent_aware_mask,\
				 need_weights=return_attn, num_agent=num_agents)

class FutureEncoder(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.nlayer = cfg.future_encoder.get('nlayer',6)
		self.num_future_frames = cfg.future_frames
		self.past_frames = cfg.past_frames
		self.context_names = cfg.get('context_names',[]) # alter this to add more context
		self.AgentformerDecoder = AgentFormerDecoder(cfg, self.nlayer)
		self.input_dim = 2*len(cfg.fut_input_type)
		for name in self.context_names:
			cfg_context = cfg.get('{}_encoder'.format(name))
			self.input_dim += cfg_context.out_dim
		self.nz = cfg.nz
		num_dist_params = 2 * self.nz
		self.d_model = cfg.get('tf_model_dim')
		self.input_fc = nn.Linear(self.input_dim, self.d_model)
		self.out_mlp_dim = cfg.future_encoder.out_mlp_dim
		self.pooling = AgentWisePooling(self.num_future_frames)
		self.q_z_net = nn.Sequential(
					MLP(self.d_model, self.out_mlp_dim, 'relu'),
					nn.Linear(self.out_mlp_dim[-1], num_dist_params)
					)
		self.time_encoder = PositionalAgentEncoding(cfg)
		self.use_agent_aware = cfg.get('use_agent_aware', True)

	def forward(self, fut_mask, future_motion, past_feature, 
		self_padding_mask, cross_padding_mask, self_agent_aware_mask=None, cross_agent_aware_mask=None, \
		context_features=None, return_attn=False):
		# selfawaremask, selfpaddingmask: [bs, futtime*agent, futtime*agent]
		# crossawaremask, crosspaddingmask: [bs, futtime*agent, pasttime*agent]
		self.max_agents = future_motion.shape[0]//self.num_future_frames
		num_agents = self.max_agents if self.use_agent_aware else 1
		tf_in = torch.cat([future_motion, context_features], dim=-1)
		tf_in = self.input_fc(tf_in)
		tf_in_timed = self.time_encoder(tf_in, num_a=self.max_agents, t_offset=self.past_frames) # t_offset 0 or 4?
		future_encoding, attn_dict = self.AgentformerDecoder(tf_in_timed, past_feature, need_weights=return_attn,
			 tgt_mask=self_padding_mask, memory_mask=cross_padding_mask, num_agent=num_agents, \
			 self_agent_aware_mask=self_agent_aware_mask, cross_agent_aware_mask=cross_agent_aware_mask)
		q_z_params = self.q_z_net(self.pooling(future_encoding, fut_mask))
		return Normal(params=q_z_params), attn_dict

class FutureDecoder(nn.Module):
	# auto regress with mlp and decoder
	def __init__(self, cfg):
		super().__init__()
		self.nlayer = cfg.future_decoder.get('nlayer', 6)
		self.d_model = cfg.tf_model_dim
		self.forecast_dim = cfg.get('forecast_dim', 2)
		self.num_future_frames = cfg.future_frames
		self.context_names = cfg.get('context_names',[]) # alter this to add extra context
		self.AgentformerDecoder = AgentFormerDecoder(cfg, self.nlayer)
		self.out_Linear = nn.Linear(self.d_model, self.forecast_dim)
		self.input_dim = 2*len(cfg.dec_input_type) + cfg.nz
		for name in self.context_names:
			cfg_context = cfg.get('{}_encoder'.format(name))
			self.input_dim += cfg_context.out_dim
		self.input_fc = nn.Linear(self.input_dim, self.d_model)
		self.time_encoder = PositionalAgentEncoding(cfg)
		self.max_agents = None
		self.concat_input = cfg.future_decoder.concat_input
		self.use_agent_aware = cfg.get('use_agent_aware', True)

	def sample(self, cur_motion, z, past_feature, num_sample, \
		self_padding_mask, cross_padding_mask, self_agent_aware_mask=None, cross_agent_aware_mask=None, 
		context_features=None):
		# we have to do sample by repeat_interleave because attention only takes in 3-dim input
		# This makes sense because in the input we have #num_sample z's spreading along dim=-2
		# So the other features should follow this pattern.
		
		self.max_agents = cur_motion.shape[0]
		num_agents = self.max_agents if self.use_agent_aware else 1
		
		context_features = context_features.repeat_interleave(num_sample,dim=-2)
		cur_motion = cur_motion.repeat_interleave(num_sample,dim=-2)
		past_feature = past_feature.repeat_interleave(num_sample, dim=-2)
		self_padding_mask = self_padding_mask.repeat_interleave(num_sample, dim=0)
		cross_padding_mask = cross_padding_mask.repeat_interleave(num_sample, dim=0)
		y_list = []
		attn_list = []
		# We need to transpose here because z from dist.sample() is of agentnum, nsample, bs, nz
		# and after reshape it into agentnum, -1, nz. The vectors on dim 1 must align with what is
		# generated above by repeat_interleave
		z = z.transpose(1,2).contiguous().view(z.shape[0], -1, z.shape[-1]) 
		fc_in = torch.cat([cur_motion, z, context_features], dim=-1)
		for i in range(self.num_future_frames):
			tf_in = self.input_fc(fc_in)
			cur_len = tf_in.shape[0]
			self_padding_mask_i = self_padding_mask[:,:cur_len,:cur_len]
			cross_padding_mask_i = cross_padding_mask[:,:cur_len,:]
			tf_in_timed = self.time_encoder(tf_in, num_a=self.max_agents, t_offset=0 if self.concat_input else i) # this is ok, mask will apply
			queried_result, attn = self.AgentformerDecoder(tf_in_timed, past_feature, 
				tgt_mask=self_padding_mask_i, memory_mask=cross_padding_mask_i, num_agent=num_agents,\
				self_agent_aware_mask=self_agent_aware_mask, cross_agent_aware_mask=cross_agent_aware_mask) 
			attn_list.append(attn)
			y_next_mu = self.out_Linear(queried_result[-self.max_agents:]) + cur_motion
			y_list.append(y_next_mu)
			f = torch.cat([y_next_mu, z, context_features], dim=-1)
			if self.concat_input:
				fc_in = torch.cat([fc_in, f],dim=0)
			else:
				fc_in = f.clone()
		y_mu = torch.stack(y_list, dim=0).transpose(0,1) # num_agents, future_frames, batchsize*nsample, forecast_dim
		y_mu = y_mu.reshape(self.max_agents, self.num_future_frames, -1, num_sample, self.forecast_dim)\
				.transpose(2,3).transpose(1,2).contiguous() # num_agents, nsample, future_frames, batchsize, forecast_dim
		return y_mu, attn_list


	def forward(self, cur_motion, z, past_feature, \
		self_padding_mask, cross_padding_mask, self_agent_aware_mask=None, cross_agent_aware_mask=None, 
		context_features=None, return_attn=False):
		self.max_agents = cur_motion.shape[0]
		num_agents = self.max_agents if self.use_agent_aware else 1
		y_list = []
		attn_list = []
		fc_in = torch.cat([cur_motion,z, context_features], dim=-1)
		for i in range(self.num_future_frames):
			tf_in = self.input_fc(fc_in)
			cur_len = tf_in.shape[0]
			self_padding_mask_i = self_padding_mask[:,:cur_len,:cur_len]
			cross_padding_mask_i = cross_padding_mask[:,:cur_len,:]
			tf_in_timed = self.time_encoder(tf_in, num_a=self.max_agents, t_offset=0 if self.concat_input else i)
			queried_result, attn = self.AgentformerDecoder(tf_in_timed, past_feature, need_weights=return_attn,
				tgt_mask=self_padding_mask_i, memory_mask=cross_padding_mask_i, num_agent=num_agents , \
				self_agent_aware_mask=self_agent_aware_mask, cross_agent_aware_mask=cross_agent_aware_mask)
			attn_list.append(attn)
			y_next_mu = self.out_Linear(queried_result[-self.max_agents:]) + cur_motion # residual learning
			y_list.append(y_next_mu)
			f = torch.cat([y_next_mu,z,context_features], dim=-1)
			if self.concat_input:
				fc_in = torch.cat([fc_in, f],dim=0)
			else:
				fc_in = f.clone()
		y_mu = torch.stack(y_list, dim=0).transpose(0,1).contiguous()
		return y_mu, attn_list


class AgentFormer(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.d_model = cfg.tf_model_dim
		self.past_frames = cfg.past_frames
		self.future_frames = cfg.future_frames
		self.max_agents = cfg.max_agent_num
		# non-sequential data we define it as context
		self.context_names = cfg.get('context_names',[]) # alter this to add extra context
		self.context_cfgs = [cfg.get('{}_encoder'.format(name)) for name in self.context_names]
		self.contextEncoders = nn.ModuleDict(
			{name: context_model_factory[name](cfg_, batch_size=cfg.batch_size, max_agent_num=cfg.max_agent_num)\
				 for name, cfg_ in zip(self.context_names, self.context_cfgs)}
			)
		self.dec_input_type = cfg.dec_input_type
		self.past_encoder = PastEncoder(cfg)

		num_dist_params = 2 * cfg.nz
		self.pooling = AgentWisePooling(self.past_frames)
		self.out_mlp_dim = cfg.future_decoder.out_mlp_dim
		self.prior_net = nn.Sequential(
					MLP(self.d_model, self.out_mlp_dim, 'relu'),
					nn.Linear(self.out_mlp_dim[-1], num_dist_params)
					)
		self.future_encoder = FutureEncoder(cfg)
		self.future_decoder = FutureDecoder(cfg)
		
		if self.cfg.map_moco:
			self.moco_mlp = nn.Linear(cfg.agent_maps_encoder.out_dim, self.d_model)

		if self.cfg.loss_cfg.clip:
			self.clip_head = nn.Linear(2*len(cfg.fut_input_type), self.d_model)
			self.traj_logist_map_mlp = nn.Linear(self.d_model, self.d_model)
			self.traj_logist_stateclass_mlp = nn.Linear(self.d_model, self.d_model)
			self.map_mlp = nn.Linear(cfg.agent_maps_encoder.out_dim, self.d_model)
			
		self.debug = cfg.get('debug', False)
		en_in_mlp_dim = cfg.future_encoder.get('in_mlp_dim')
		if en_in_mlp_dim is not None:
			self.encoder_input_mlp = nn.Sequential(
				MLP(self.d_model, en_in_mlp_dim, 'relu'),
				nn.Linear(en_in_mlp_dim[-1], self.d_model)
				)
		self.en_in_mlp_dim = en_in_mlp_dim
		
		de_in_mlp_dim = cfg.future_decoder.get('in_mlp_dim')
		if de_in_mlp_dim is not None:
			self.decoder_input_mlp = nn.Sequential(
				MLP(self.d_model, de_in_mlp_dim, 'relu'),
				nn.Linear(de_in_mlp_dim[-1], self.d_model)
				)
		self.de_in_mlp_dim = de_in_mlp_dim

	def inference(self, data, num_samples=1, return_attn=False):
		context_dict = {name:data[name] for name in self.context_names}
		context_features_list = extract_and_aggregate_context(context_dict, 
			self.contextEncoders, self.max_agents)
		context_features = torch.cat(context_features_list, dim=-1)
		past_context_features = context_features.repeat(self.past_frames, 1, 1)
		past_bef = data['pre_input']
		
		past_feature, past_attn = self.past_encoder(past_bef, context_features=past_context_features, \
				padding_mask=data['pastEn_self_padding_mask'], return_attn=return_attn)
		p_z = Normal(params=self.prior_net(self.pooling(past_feature, data['pre_mask'])))
		z = p_z.sample(num_samples)

		if 'heading' in self.dec_input_type:
			fut_dec_context_features = torch.cat([context_features, data['heading_vec']], dim=-1)

		fut_pred, attn = self.future_decoder.sample(data['cur_input'], z, past_feature, num_sample=num_samples, \
				context_features=fut_dec_context_features, \
				self_padding_mask=data['futEn_self_padding_mask'],cross_padding_mask=data['futEn_cross_padding_mask'])
		return fut_pred, attn

	def forward(self, data, return_attn=False):
		context_dict = {name:data[name] for name in self.context_names}
		clip_dict = {}
		context_feature_list = extract_and_aggregate_context(context_dict, 
			self.contextEncoders, self.max_agents)
		# repeat the (agent num, bs, context_dim) into (Txagent num, bs, context_dim)
		context_features = torch.cat(context_feature_list, dim=-1)
		past_context_features = context_features.repeat(self.past_frames, 1, 1)
		# print(data['state'].shape, data['class'].shape)
		past_bef = data['pre_input']
		future_bef = data['future_input']
		
		past_feature, attn_past_enc = self.past_encoder(past_bef, context_features=past_context_features, \
				padding_mask=data['pastEn_self_padding_mask'], return_attn=return_attn)
		
		p_z = Normal(params=self.prior_net(self.pooling(past_feature, data['pre_mask'])))

		if self.cfg.map_moco and self.training:
			orig_map = self.contextEncoders['agent_maps'](data['agent_maps_moco'])
			augments = self.contextEncoders['agent_maps'](data['agent_maps_moco_augment'])
			orig_map = self.moco_mlp(orig_map)
			augments = self.moco_mlp(augments)
			clip_dict['orig_map_logist'] = orig_map / orig_map.norm(dim=-1, keepdim=True)
			clip_dict['augments_logist'] = augments / augments.norm(dim=-1, keepdim=True)
			
		if self.cfg.loss_cfg.clip.do and self.training:
			f_dict = {}
			for i in range(len(context_dict.keys())):
				f = context_feature_list[i].transpose(0, 1).contiguous()
				f_dict[list(context_dict.keys())[i]] = f

			map_f = self.map_mlp(f_dict['agent_maps'])
			clip_dict['agent_maps_logist'] = map_f / map_f.norm(dim=-1, keepdim=True)

			if self.cfg.loss_cfg.clip.pastenc and not self.cfg.use_traj_enc:
				past_bef = self.clip_head(past_bef)
				past_bef, attn_past_enc = self.past_encoder(past_bef, context_features=None, \
					padding_mask=data['pastEn_self_padding_mask'], return_attn=return_attn)
			past_bef = self.clip_head(past_bef)	
			past_feature_traj = past_bef.view(
				self.past_frames, -1, past_bef.shape[-2], past_bef.shape[-1]
				).mean(dim=0)
			
			traj_logist =  self.traj_logist_map_mlp(past_feature_traj.permute(1,0,2))
			clip_dict['traj_map_logist'] = traj_logist / (traj_logist.norm(dim=-1, keepdim=True) + 1e-5)

			if self.cfg.loss_cfg.clip.pretrain:
				return None, None, None, clip_dict
		
		if self.en_in_mlp_dim is not None:
			past_feature_transformed = self.encoder_input_mlp(past_feature)
		else:
			past_feature_transformed = past_feature
		fut_context_features = context_features.repeat(self.future_frames, 1, 1)
		
		q_z, _ = self.future_encoder(data['fut_mask'], future_bef, past_feature_transformed, context_features=fut_context_features, \
			self_padding_mask=data['futEn_self_padding_mask'],cross_padding_mask=data['futEn_cross_padding_mask'])
		z = q_z.rsample()

		if self.de_in_mlp_dim is not None:
			past_feature_transformed = self.decoder_input_mlp(past_feature)
		else:
			past_feature_transformed = past_feature
		if 'heading' in self.dec_input_type:
			fut_dec_context_features = torch.cat([context_features, data['heading_vec']], dim=-1)
		fut_pred, _ = self.future_decoder(data['cur_input'], z, past_feature_transformed, context_features=fut_dec_context_features, \
			self_padding_mask=data['futEn_self_padding_mask'],cross_padding_mask=data['futEn_cross_padding_mask'])
		
		return fut_pred, p_z, q_z, clip_dict



