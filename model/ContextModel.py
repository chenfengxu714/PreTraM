import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .resnet import resnet18, resnet34, resnet50

class FlexibleTensor():
    def __init__(self, x, postprocess_op='repeat_interleave',**kwargs):
        self.x = x 
        self.option = kwargs
        self.postprocess_op = 'repeat_interleave'

    def postprocess(self, **kwargs):
        if self.postprocess_op=='repeat_interleave':
            repeat_time = kwargs['n_frames']
            return self.x.repeat_interleave(n_frames, dim=0)


class MapCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg is not None
        self.convs = nn.ModuleList()
        map_channels = cfg.get('map_channels', 3)
        patch_size = cfg.get('patch_size', [100, 100])
        hdim = cfg.get('hdim', [32, 32])
        kernels = cfg.get('kernels', [3, 3])
        strides = cfg.get('strides', [3, 3])
        self.out_dim = out_dim = cfg.get('out_dim', 32)
        self.input_size = input_size = (map_channels, patch_size[0], patch_size[1])
        x_dummy = torch.randn(input_size).unsqueeze(0)

        for i, _ in enumerate(hdim):
            self.convs.append(nn.Conv2d(map_channels if i == 0 else hdim[i-1],
                                        hdim[i], kernels[i],
                                        stride=strides[i]))
            x_dummy = self.convs[i](x_dummy)

        self.fc = nn.Linear(x_dummy.numel(), out_dim)
        self.dropout = nn.Dropout(cfg.get('dropout', 0.1))
    def forward(self, x):
        for conv in self.convs:
            x = self.dropout(F.leaky_relu(conv(x), 0.2))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

class MapEncoder(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        model_id = cfg.get('model_id', 'map_cnn')
        dropout = cfg.get('dropout', 0.0)
        self.normalize = cfg.get('normalize', True)
        self.dropout = nn.Dropout(dropout)
        if model_id == 'map_cnn':
            self.model = MapCNN(cfg)
            self.out_dim = self.model.out_dim
        elif 'resnet' in model_id:
            model_dict = {
                'resnet18': resnet18,
                'resnet34': resnet34,
                'resnet50': resnet50
            }
            self.out_dim = out_dim = cfg.get('out_dim', 32)
            self.model = model_dict[model_id](pretrained=False, norm_layer=nn.InstanceNorm2d, dropout=cfg.get('dropout', 0.1))
            self.model.fc = nn.Linear(self.model.fc.in_features, out_dim)
        else:
            raise ValueError('unknown map encoder!')

    def forward(self, x):
        if self.normalize:
            x = x * 2. - 1.
        x = self.model(x)
        x = self.dropout(x)
        return x

class AgentMapEncoder(MapEncoder):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self.bsz = kwargs['batch_size']

    def forward(self, x):
        bsz, max_agent, C, H, W = x.shape
        x = super().forward(x.view(bsz*max_agent, C, H, W))
        x = x.reshape(-1, max_agent, x.shape[-1]).transpose(0,1).contiguous()
        return x

        # bsz = self.bsz if self.training else 1  #chenfeng hard code here for evaluation
        # max_agents = x.shape[0] // bsz
        # x = super().forward(x)
        # x = x.reshape(bsz, max_agents, -1).transpose(0,1).contiguous()
        # return x

class AgentStateEncoder(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        input_dim = cfg.get('state_num', 8)
        out_dim = cfg.get('out_dim', 32)
        self.cfg = cfg
        if cfg.get('embedding', True):
            self.state_embedding = nn.Embedding(input_dim, out_dim)
        else:
            self.state_encoder = nn.Sequential(
                nn.Conv1d(input_dim, 32, 1),
                nn.BatchNorm1d(32),
                nn.ReLU(True),
                nn.Conv1d(32, out_dim, 1),
                nn.BatchNorm1d(out_dim),
                )
            
    def forward(self, x):
        if self.cfg.get('embedding', True):
            idx = x.argmax(dim=-1).long()
            embed = self.state_embedding(idx)
            # print(embed.shape, 'state')
            return embed.permute(1,0,2).contiguous()
        else:
            return self.state_encoder(x.transpose(1,2)).permute(2,0,1).contiguous()
        

class AgentClassEncoder(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        input_dim = cfg.get('class_num', 18)
        out_dim = cfg.get('out_dim', 32)
        self.cfg = cfg
        if cfg.get('embedding', True):
            self.class_embedding = nn.Embedding(input_dim, out_dim)
        
        else:
            self.class_encoder = nn.Sequential(
                nn.Conv1d(input_dim, 32, 1),
                nn.BatchNorm1d(32),
                nn.ReLU(True),
                nn.Conv1d(32, out_dim, 1),
                nn.BatchNorm1d(out_dim),
            )
            
    def forward(self, x): # x: batch size, agent num, inputdim
        if self.cfg.get('embedding', True):
            bs, an, d = x.shape
            idx = x.argmax(dim=-1).long()
            embed = self.class_embedding(idx)
            # print(embed.shape, 'class')
            return embed.permute(1,0,2).contiguous()
        else:
            return self.class_encoder(x.transpose(1,2)).permute(2,0,1).contiguous()
        

context_model_factory = {
	'map': MapEncoder,
    'agent_maps': AgentMapEncoder,
    "state": AgentStateEncoder,
    "class": AgentClassEncoder
}