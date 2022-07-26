import torch
import torch.nn as nn
class AgentWisePooling(nn.Module):
	def __init__(self, num_frames):
		super(AgentWisePooling, self).__init__()
		self.num_frames = num_frames 

	def forward(self, x, mask=None):
		if mask is not None:
			mask = mask.transpose(0, 1).contiguous().view(-1, x.shape[1])
			mask = mask.unsqueeze(-1)
			x = x * mask
		x = x.reshape(self.num_frames, -1, x.shape[-2], x.shape[-1])
		return torch.max(x, dim=0)[0]