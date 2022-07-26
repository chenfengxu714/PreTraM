from transformers import get_linear_schedule_with_warmup
from torch.optim import lr_scheduler

def get_scheduler(optimizer, policy, **kwargs):
	if policy == 'lambda':
		raise NotImplementedError
		def lambda_rule(epoch):
			lr_l = 1.0 - max(0, epoch - kwargs['nepoch_fix']) / float(kwargs['nepoch'] - kwargs['nepoch_fix'] + 1)
			return lr_l
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif policy == 'step':
		scheduler = lr_scheduler.StepLR(
			optimizer, step_size=kwargs['decay_step'], gamma=kwargs['decay_gamma'])
	elif policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(
			optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
	elif policy == 'warmup':
		scheduler = get_linear_schedule_with_warmup(optimizer, 
							num_warmup_steps=kwargs['warmup_rate']*kwargs['total_steps'],
							num_training_steps=kwargs['total_steps'])
	else:
		return NotImplementedError('learning rate policy [%s] is not implemented', policy)
	return scheduler


class LR_Scheduler():
	def __init__(self, cfg, optimizer, steps_per_update=1, steps_per_epoch=1):
		self.steps_per_update = steps_per_update
		self.cnt_steps = 0
		self.scheduler = get_scheduler(optimizer, policy=cfg.lr_scheduler, decay_step=cfg.decay_step, \
			decay_gamma=cfg.decay_gamma, warmup_rate=cfg.warmup_rate, total_steps=cfg.num_epochs*steps_per_epoch) 
		self.name = cfg.lr_scheduler

	def step(self):
		self.cnt_steps += 1
		if self.cnt_steps % self.steps_per_update == 0:
			self.scheduler.step()
			return

	def state_dict(self):
		return self.scheduler.state_dict()

