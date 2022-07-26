import sys, os
from tqdm import tqdm
from .base_runner import BaseEvaler
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.utils import AverageMeter
from utils.utils import prepare_seed, print_log, mkdir_if_missing

class AgentFormerEvaler(BaseEvaler):
	def __init__(self, cfg, model, device, data_loader, save_dir, split, log):
		super().__init__(model, device, data_loader, save_dir, split, log)
		self.model = model
		self.n_samples = cfg.sample_k
		self.traj_scale = cfg.traj_scale
		self.save_dir = save_dir
		self.profiling = cfg.get('profiling', False)
		self.total_iter = len(data_loader)
		self.cfg = cfg
		self.split = split
		self.cur_batch = None
		self.device = device
		self.log = log


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
		
		data['state'] = data['state'].float().to(self.device)
		data['class'] = data['class'].float().to(self.device)
		
		data['agent_maps'] = agent_maps.reshape(bsz, n_agent, -1, height, width).to(self.device)
		if 'heading' in self.cfg.dec_input_type:
			data['heading_vec'] = data['heading_vec'].transpose(0,1).contiguous().to(self.device)
			# data['cur_motion'] = data['cur_motion'].reshape(bsz, n_agent, -1).transpose(0,1).contiguous()
		data['fut_motion_orig'] = data['fut_motion_orig'].transpose(0,1).transpose(1,2).contiguous().to(self.device)
		data['scene_orig'] = data['scene_orig'].unsqueeze(0).unsqueeze(1).to(self.device)
		data['pastEn_self_padding_mask'] = data['pastEn_self_padding_mask'].to(self.device)
		data['futEn_self_padding_mask'] = data['futEn_self_padding_mask'].to(self.device)
		data['futEn_cross_padding_mask'] = data['futEn_cross_padding_mask'].to(self.device)
		self.cur_batch = data

	
	def get_model_prediction(self, data, sample_k):
		self.prepare_data(data)
		sample_motion_3D, data = self.model.inference(data, self.n_samples)
		return sample_motion_3D
		
	def evaluation(self, data):
		# gt_motion_3D = data['future_input'] * self.traj_scale
		sample_motion_3D = self.get_model_prediction(data, self.n_samples) 
		sample_motion_3D = self.postprocessing(sample_motion_3D)
		sample_motion_3D = sample_motion_3D * self.traj_scale
		return sample_motion_3D
		
	def save_samples(self, gt_motion_3D, sample_motion_3D, data):
		gt_motion_3D = self.traj_scale * gt_motion_3D
		gt_motion_3D = gt_motion_3D.squeeze(dim=0).transpose(0, 1) # for batch size = 1, then becomes (agentnum, T, 2)
		sample_dir = os.path.join(self.save_dir, 'samples'); mkdir_if_missing(sample_dir)
		gt_dir = os.path.join(self.save_dir, 'gt'); mkdir_if_missing(gt_dir)
		sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
		sample_motion_3D = sample_motion_3D.squeeze(dim=3) # for batch size = 1 (sample_num, agent_num, T, 2)
		for i in range(sample_motion_3D.shape[0]):
			self.save_prediction(sample_motion_3D[i], data, f'/sample_{i:03d}', sample_dir)
		num_pred = self.save_prediction(gt_motion_3D, data, '', gt_dir)              # save gt
		return num_pred

	def save_prediction(self, pred, data, suffix, save_dir):
		pred_num = 0
		pred_arr = []
		fut_data, seq_name, frame, valid_id, pred_mask = data['fut_data'], data['seq'], data['frame'], data['valid_id'], data['pred_mask']
		assert pred_mask is not list
		pred_mask = pred_mask[0]
		seq_name = seq_name[0]
		frame = frame[0]
		# fut_data : a list with len future_frames [(1, 1, 18)] 
		for i in range(len(valid_id[0])):    # number of agents
			# print(valid_id[0])
			identity = valid_id[0][i]
			if pred_mask is not None and pred_mask[i] != 1.0:
				continue
			for j in range(self.cfg.future_frames):
				cur_data = fut_data[0][j] #also squeeze for batch size = 1, be careful here.
				if len(cur_data) > 0 and identity in cur_data[:, 1]:
					data = cur_data[cur_data[:, 1] == identity].squeeze()
				else:
					data = most_recent_data.copy()
					data[0] = frame + j + 1
				data[[13, 15]] = pred[i, j].cpu().numpy()  # [13, 15] corresponds to 2D pos
				most_recent_data = data.copy()
				pred_arr.append(data)
			pred_num += 1

		if len(pred_arr) > 0:
			pred_arr = np.vstack(pred_arr)
			indices = [0, 1, 13, 15]            # frame, ID, x, z (remove y which is the height)
			pred_arr = pred_arr[:, indices]
			fname = f'{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt'
			mkdir_if_missing(fname)
			np.savetxt(fname, pred_arr, fmt="%.3f")
		return pred_num


	def postprocessing(self, fut_motion_samples):
		if self.cfg.pred_type=='orig':
		# in world coordinate  
			pass
		elif self.cfg.pred_type=='scene_norm': # convert to world coordinate
			fut_motion_samples += self.cur_batch['scene_orig']
		elif self.cfg.pred_type=='vel': # TODO add velocity
			raise NotImplementedError
		else:
			raise NotImplementedError
		return fut_motion_samples


class DLowEvaler(BaseEvaler):
	# need to specify: train_iteration, compute_loss, generate_msg
	def __init__(self, cfg, model, device, data_loader, save_dir, split, log):
		super().__init__(model, device, data_loader, save_dir, split, log)
		self.traj_scale = cfg.traj_scale
		self.save_dir = save_dir
		self.split = split
		self.log = log
		self.model = model
		self.pre_cfg = self.model.pre_model.cfg
		self.n_samples = cfg.sample_k
		self.profiling = cfg.get('profiling', False)
		self.total_iter = len(data_loader)
		self.cfg = cfg
		self.cur_batch = None
		self.device = device
		self.agentFormerEvaler = AgentFormerEvaler(self.pre_cfg, model, device, data_loader, save_dir, split, log)

	def prepare_data(self, data):
		self.agentFormerEvaler.prepare_data(data)
		data['agent_num']=data['agent_num'].to(self.device)
		self.cur_batch = data
	
	def get_model_prediction(self, data, sample_k):
		self.prepare_data(data)
		fut_motion_samples, p_z, sample_z = self.model(data)
		fut_motion_samples, p_z, sample_z = self.postprocessing(fut_motion_samples, p_z, sample_z)
		return fut_motion_samples
		
	def evaluation(self, data):
		sample_motion_3D = self.get_model_prediction(data, self.n_samples) 
		sample_motion_3D = sample_motion_3D * self.traj_scale
		return sample_motion_3D
		
	def save_samples(self, gt_motion_3D, sample_motion_3D, data):
		gt_motion_3D = self.traj_scale * gt_motion_3D
		gt_motion_3D = gt_motion_3D.squeeze(dim=0).transpose(0, 1) # for batch size = 1, then becomes (agentnum, T, 2)
		sample_dir = os.path.join(self.save_dir, 'samples'); mkdir_if_missing(sample_dir)
		gt_dir = os.path.join(self.save_dir, 'gt'); mkdir_if_missing(gt_dir)
		sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
		sample_motion_3D = sample_motion_3D.squeeze(dim=3) # for batch size = 1 (sample_num, agent_num, T, 2)
		for i in range(sample_motion_3D.shape[0]):
			self.save_prediction(sample_motion_3D[i], data, f'/sample_{i:03d}', sample_dir)
		num_pred = self.save_prediction(gt_motion_3D, data, '', gt_dir)              # save gt
		return num_pred

	def save_prediction(self, pred, data, suffix, save_dir):
		pred_num = 0
		pred_arr = []
		fut_data, seq_name, frame, valid_id, pred_mask = data['fut_data'], data['seq'], data['frame'], data['valid_id'], data['pred_mask']
		assert pred_mask is not list
		pred_mask = pred_mask[0]
		seq_name = seq_name[0]
		frame = frame[0] 
		for i in range(len(valid_id[0])):    # number of agents
			identity = valid_id[0][i]
			if pred_mask is not None and pred_mask[i] != 1.0:
				continue
			for j in range(self.cfg.future_frames):
				cur_data = fut_data[0][j] #also squeeze for batch size = 1, be careful here.
				if len(cur_data) > 0 and identity in cur_data[:, 1]:
					data = cur_data[cur_data[:, 1] == identity].squeeze()
				else:
					data = most_recent_data.copy()
					data[0] = frame + j + 1
				data[[13, 15]] = pred[i, j].cpu().numpy()  # [13, 15] corresponds to 2D pos
				most_recent_data = data.copy()
				pred_arr.append(data)
			pred_num += 1

		if len(pred_arr) > 0:
			pred_arr = np.vstack(pred_arr)
			indices = [0, 1, 13, 15]            # frame, ID, x, z (remove y which is the height)
			pred_arr = pred_arr[:, indices]
			fname = f'{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt'
			mkdir_if_missing(fname)
			np.savetxt(fname, pred_arr, fmt="%.3f")
		return pred_num

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

evaler_factory = {
	'agentformer': AgentFormerEvaler,
	'dlow': DLowEvaler
}
