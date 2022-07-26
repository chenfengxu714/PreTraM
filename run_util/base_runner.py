import torch
from tqdm import tqdm
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.utils import AverageMeter
from utils.utils import prepare_seed, print_log, mkdir_if_missing

class BaseRunner(object):
	def __init__(self, model, data_loader):
		self.model = model
		self.data_loader = data_loader

class BaseTrainer(BaseRunner):
	def __init__(self, model, data_loader, optimizer, lr_scheduler, logger=None, print_freq=100):
		super().__init__(model, data_loader)
		self.optimizer = optimizer
		self.lr_scheduler = lr_scheduler 
		self.print_freq = print_freq
		self.total_loss_meter = AverageMeter()
		self.logger = logger

	def train_epoch(self):

		self.total_loss_meter.reset()
		for k in self.loss_meters:
			self.loss_meters[k].reset()
		if self.profiling:
			self.train_epoch_profiling()
			return
		len_data = len(self.data_loader)
		for i, data in enumerate(tqdm(self.data_loader)):
			output = self.train_iteration(data)
			total_loss = self.compute_loss(output, data)
			self.optimizer.zero_grad()
			total_loss.backward()
			self.optimizer.step()
			if i%self.print_freq ==0:
				msg = self.generate_msg(i)
				if self.logger is not None:
					self.logger.info(msg)
				else:
					print(msg)
			if 'warmup' in self.lr_scheduler.name:
				self.lr_scheduler.step()
		if not 'warmup' in self.lr_scheduler.name:
			self.lr_scheduler.step()
		
		return self.total_loss_meter.avg

	def save_state(self, log_dir, epoch, best=False):
		state_dict = {'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict(),\
		 'scheduler':self.lr_scheduler.state_dict(), 'epoch': epoch+1} # a model loading this state dict should start from epoch+1
		if best:
			with open(os.path.join(log_dir, 'model_best.pth'),'wb') as f:
				torch.save(state_dict,f)
		else:
			with open(os.path.join(log_dir, 'model_'+str(epoch)+'.pth'),'wb') as f:
				torch.save(state_dict,f)

	def train_epoch_profiling(self):
		pass


class BaseEvaler(BaseRunner):
	def __init__(self, model, device, data_loader, save_dir=None, split="val", log=None):
		super().__init__(model, data_loader)
		self.save_dir = save_dir
		self.split = split
		self.device = device
		self.log = log
	
	def test_model(self):
		total_num_pred = 0
		len_data = len(self.data_loader)
		test_list = []
		for i, data in enumerate(tqdm(self.data_loader)):
			if data is None:
				continue
			seq_name, frame = data['seq'], data['frame']
			if (seq_name, frame) in test_list:
				continue
			else:
				test_list.append((seq_name, frame))
				gt_motion_3D = data['fut_motion'].to(self.device)
				with torch.no_grad():
					sample_motion_3D = self.evaluation(data)
				num_pred = self.save_samples(gt_motion_3D, sample_motion_3D, data)
				total_num_pred += num_pred
		
		print_log(f'\n\n total_num_pred: {total_num_pred}', self.log)



