from .agentformer import AgentFormer
from .loss import loss_factory
from .dlow import *
from .resnet import *
model_factory = {
	'agentformer': AgentFormer,
	'dlow': DLow
}