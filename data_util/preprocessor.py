import torch, os, numpy as np, copy
import torch.nn.functional as F
import cv2
import glob
from .map import GeometricMap
import random

class preprocess(object):
    
    def __init__(self, data_root, seq_name, parser, log, split='train', phase='training'):
        self.parser = parser
        self.dynamic_padding = parser.get('dynamic_padding', False)
        self.dataset = parser.dataset
        self.data_root = data_root
        self.past_frames = parser.past_frames
        self.future_frames = parser.future_frames
        self.frame_skip = parser.get('frame_skip', 1)
        self.min_past_frames = parser.get('min_past_frames', self.past_frames)
        self.min_future_frames = parser.get('min_future_frames', self.future_frames)
        self.traj_scale = parser.traj_scale
        self.past_traj_scale = parser.traj_scale
        self.load_map = parser.get('load_map', False)
        self.map_moco = parser.get('map_moco', False)
        self.moco_num = parser.get('moco_num', 8)
        self.map_version = parser.get('map_version', '0.1')
        self.seq_name = seq_name
        self.split = split
        self.phase = phase
        self.log = log
        self.max_agent_num = parser.max_agent_num
        self.use_map = parser.get('use_map', False)
        self.rand_rot_scene = parser.get('rand_rot_scene', False)
        self.discrete_rot = parser.get('discrete_rot', False)
        self.map_global_rot = parser.get('map_global_rot', False)
        self.ar_train = parser.get('ar_train', True)
        self.agent_enc_shuffle = parser.get('agent_enc_shuffle', False)
        self.input_type = parser.get('input_type', 'pos')
        self.pred_type = parser.get('pred_type', self.input_type)
        self.fut_input_type = parser.get('fut_input_type', self.input_type)
        self.dec_input_type = parser.get('dec_input_type', [])
        self.vel_heading = parser.get('vel_heading', False)
        self.one_direction_past=parser.get('one_direction_past', False)

        if parser.dataset == 'nuscenes_pred':
            label_path = os.path.join(data_root, 'label/{}/{}.txt'.format(split, seq_name))
            descr_path = os.path.join(data_root, 'descr/{}/{}.txt'.format(split, seq_name))
            delimiter = ' '
        elif parser.dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
            label_path = f'{data_root}/{parser.dataset}/{seq_name}.txt'
            delimiter = ' '
        else:
            assert False, 'error'

        self.gt = np.genfromtxt(label_path, delimiter=delimiter, dtype=str)
        self.description = np.genfromtxt(descr_path, delimiter=delimiter, dtype=str)
        
        frames = self.gt[:, 0].astype(np.float32).astype(np.int)
        fr_start, fr_end = frames.min(), frames.max()
        self.init_frame = fr_start
        self.num_fr = fr_end + 1 - fr_start
        # print(self.num_fr)
        if self.load_map:
            self.load_scene_map()
        else:
            self.geom_scene_map = None

        self.class_names = class_names = {'Pedestrian': 1, 'Car': 2, 'Cyclist': 3, 'Truck': 4, 'Van': 5, 'Tram': 6, 'Person': 7, \
            'Misc': 8, 'DontCare': 9, 'Traffic_cone': 10, 'Construction_vehicle': 11, 'Barrier': 12, 'Motorcycle': 13, \
            'Bicycle': 14, 'Bus': 15, 'Trailer': 16, 'Emergency': 17, 'Construction': 18}
        
        self.one_hot_state = {'None': 0, 'Moving': 1, "Stopped": 2,
            'Parked': 3, 'Withrider': 4, "Withoutrider": 5,
            "Sitting": 6, 'Standing': 7
            }
        self.state_num = len(self.one_hot_state)
        self.class_num = len(self.class_names)
        
        for row_index in range(len(self.gt)):
            self.gt[row_index][2] = class_names[self.gt[row_index][2]]
            self.gt[row_index][3] = self.one_hot_state[self.gt[row_index][3]]
            
        self.gt = self.gt.astype('float32')
        self.xind, self.zind = 13, 15

    def GetID(self, data):
        id = []
        for i in range(data.shape[0]):
            id.append(data[i, 1].copy())
        return id

    def TotalFrame(self):
        return self.num_fr

    def PreData(self, frame):
        DataList = []
        for i in range(self.past_frames):
            if frame - i < self.init_frame:              
                data = []
            data = self.gt[self.gt[:, 0] == (frame - i * self.frame_skip)]    
            DataList.append(data)
        return DataList
    
    def FutureData(self, frame):
        DataList = []
        for i in range(1, self.future_frames + 1):
            data = self.gt[self.gt[:, 0] == (frame + i * self.frame_skip)]
            DataList.append(data)
        return DataList

    def get_valid_id(self, pre_data, fut_data):
        cur_id = self.GetID(pre_data[0])
        valid_id = []
        for idx in cur_id:
            exist_pre = [(False if isinstance(data, list) else (idx in data[:, 1])) for data in pre_data[:self.min_past_frames]]
            exist_fut = [(False if isinstance(data, list) else (idx in data[:, 1])) for data in fut_data[:self.min_future_frames]]
            if np.all(exist_pre) and np.all(exist_fut):
                valid_id.append(idx)
        return valid_id
    
    def get_one_hot_state(self, cur_data, valid_id):
        one_hot_state = np.zeros((len(valid_id), self.state_num))
        for i, idx in enumerate(valid_id):
            state_id = cur_data[cur_data[:, 1] == idx].squeeze()[3]
            state_onehot = (np.arange(self.state_num) == state_id).astype(np.float32)
            one_hot_state[i, :] = state_onehot
        return one_hot_state
        
    def get_one_hot_class(self, cur_data, valid_id):
        one_hot_class = np.zeros((len(valid_id), self.class_num))
        for i, idx in enumerate(valid_id):
            class_id = cur_data[cur_data[:, 1] == idx].squeeze()[2]
            class_onehot = (np.arange(self.class_num) == class_id).astype(np.float32)
            one_hot_class[i, :] = class_onehot
        return one_hot_class
    
    def get_pred_mask(self, cur_data, valid_id):
        pred_mask = np.zeros(len(valid_id), dtype=np.int)
        for i, idx in enumerate(valid_id):
            pred_mask[i] = cur_data[cur_data[:, 1] == idx].squeeze()[-1]
        return pred_mask

    def get_heading(self, cur_data, valid_id):
        heading = np.zeros(len(valid_id))
        for i, idx in enumerate(valid_id):
            heading[i] = cur_data[cur_data[:, 1] == idx].squeeze()[16]
        return heading

    def load_scene_map(self):
        map_file = f'{self.data_root}/map_{self.map_version}/{self.seq_name}.png'
        map_vis_file = f'{self.data_root}/map_{self.map_version}/vis_{self.seq_name}.png'
        map_meta_file = f'{self.data_root}/map_{self.map_version}/meta_{self.seq_name}.txt'
        self.scene_map = np.transpose(cv2.imread(map_file), (2, 0, 1))
        self.scene_vis_map = np.transpose(cv2.cvtColor(cv2.imread(map_vis_file), cv2.COLOR_BGR2RGB), (2, 0, 1))
        self.meta = np.loadtxt(map_meta_file)
        self.map_origin = self.meta[:2]
        self.map_scale = scale = self.meta[2]
        homography = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., scale]])
        self.geom_scene_map = GeometricMap(self.scene_map, homography, self.map_origin)
        self.scene_vis_map = GeometricMap(self.scene_vis_map, homography, self.map_origin)

    def PreMotion(self, DataTuple, valid_id):
        motion = []
        mask = []
        for identity in valid_id:
            mask_i = torch.zeros(self.past_frames)
            box_3d = torch.zeros([self.past_frames, 2])
            for j in range(self.past_frames):
                past_data = DataTuple[j]              # past_data
                if len(past_data) > 0 and identity in past_data[:, 1]:
                    found_data = past_data[past_data[:, 1] == identity].squeeze()[[self.xind, self.zind]] / self.past_traj_scale
                    box_3d[self.past_frames-1 - j, :] = torch.from_numpy(found_data).float()
                    mask_i[self.past_frames-1 - j] = 1.0
                elif j > 0:
                    box_3d[self.past_frames-1 - j, :] = box_3d[self.past_frames - j, :]    # if none, copy from previous
                else:
                    raise ValueError('current id missing in the first frame!')
            motion.append(box_3d)
            mask.append(mask_i)
        return motion, mask

    def FutureMotion(self, DataTuple, valid_id):
        motion = []
        mask = []
        for identity in valid_id:
            mask_i = torch.zeros(self.future_frames)
            pos_3d = torch.zeros([self.future_frames, 2])
            for j in range(self.future_frames):
                fut_data = DataTuple[j]              # cur_data
                if len(fut_data) > 0 and identity in fut_data[:, 1]:
                    found_data = fut_data[fut_data[:, 1] == identity].squeeze()[[self.xind, self.zind]] / self.traj_scale
                    pos_3d[j, :] = torch.from_numpy(found_data).float()
                    mask_i[j] = 1.0
                elif j > 0:
                    pos_3d[j, :] = pos_3d[j - 1, :]    # if none, copy from previous
                else:
                    raise ValueError('current id missing in the first frame!')
            motion.append(pos_3d)
            mask.append(mask_i)
        return motion, mask
    
    def rotation_2d_torch(self, x, theta, origin=None):
        if origin is None:
            origin = torch.zeros(2).to(x.device).to(x.dtype)
        norm_x = x - origin
        norm_rot_x = torch.zeros_like(x)
        norm_rot_x[..., 0] = norm_x[..., 0] * torch.cos(theta) - norm_x[..., 1] * torch.sin(theta)
        norm_rot_x[..., 1] = norm_x[..., 0] * torch.sin(theta) + norm_x[..., 1] * torch.cos(theta)
        rot_x = norm_rot_x + origin
        return rot_x, norm_rot_x
    
    def pre_input_generate(self, data):
        traj_in = []
        for key in self.input_type:
            if key == 'pos':
                traj_in.append(data['pre_motion'])
            elif key == 'vel':
                vel = data['pre_vel']
                if len(self.input_type) > 1:
                    vel = torch.cat([vel[[0]], vel], dim=0)
                if self.vel_heading:
                    vel = rotation_2d_torch(vel, -data['heading'])[0]
                traj_in.append(vel)
            elif key == 'norm':
                traj_in.append(data['pre_motion_norm'])
            elif key == 'scene_norm':
                traj_in.append(data['pre_motion_scene_norm'])
            elif key == 'heading':
                hv = data['heading_vec'].unsqueeze(0).repeat((data['pre_motion'].shape[0], 1, 1))
                traj_in.append(hv)
            elif key == 'map':
                continue
            else:
                raise ValueError('unknown input_type!')
        traj_in = torch.cat(traj_in, dim=-1)
        # print(data['pre_motion_scene_norm'])
        return traj_in
    

    def future_input_generate(self, data):
        traj_in = []
        for key in self.fut_input_type:
            if key == 'pos':
                traj_in.append(data['fut_motion'])
            elif key == 'vel':
                vel = data['fut_vel']
                if self.vel_heading:
                    vel = rotation_2d_torch(vel, -data['heading'])[0]
                traj_in.append(vel)
            elif key == 'norm':
                traj_in.append(data['fut_motion_norm'])
            elif key == 'scene_norm':
                traj_in.append(data['fut_motion_scene_norm'])
            elif key == 'heading':
                hv = data['heading_vec'].unsqueeze(0).repeat((data['fut_motion'].shape[0], 1, 1))
                traj_in.append(hv)
            elif key == 'map':
                continue
            else:
                raise ValueError('unknown input_type!')
        traj_in = torch.cat(traj_in, dim=-1)
        return traj_in
        
    def cur_input_generate(self, data, sample_num=1):
        traj_in = [data['pre_motion_scene_norm'][[-1]].squeeze(0)]
        for key in self.dec_input_type:
            if key == 'scene_norm':
                pass
            elif key == 'heading':
                # edited by TL, given that heading is non-sequential variable like map,
                # This is concat to context feature (map) before future decoder
                # We need to leave this out because it should be concatenated to dec_in at each
                # iterative step in the autoregressive decoder.
                continue
                # heading = data['heading_vec']
                # traj_in.append(heading)
            elif key == 'map':
                continue
            else:
                raise ValueError('wrong decode input type!')
        output = torch.cat(traj_in, dim=-1)
        return output
       
    
    def pre_self_mask_generate(self, data, mask):
        pre_mask = data['pre_mask'].view((-1, 1))
        mask_list = pre_mask
        past_mask = mask.repeat((self.past_frames, self.past_frames))
        # print(past_mask)
        if self.one_direction_past:
            tmp = np.arange(self.past_frames)
            x, y = np.meshgrid(tmp, tmp)
            direction_mask_time = torch.from_numpy(x>y)
            num_agent = pre_mask.shape[0]//self.past_frames
            direction_mask = direction_mask_time.repeat_interleave(num_agent, dim=0)\
                        .repeat_interleave(num_agent, dim=1)
            past_mask |= direction_mask
        return ((mask_list).mm(mask_list.T) == 0) | past_mask.bool()
    
    def futenc_self_mask_generate(self, data, mask):
        fut_mask = data['fut_mask'].view((-1, 1))
        mask_list = fut_mask
        future_mask = mask.repeat((self.future_frames, self.future_frames))
        return ((mask_list).mm(mask_list.T) == 0) | future_mask.bool()

    def futenc_cross_mask_generate(self, data, mask):
        pre_mask = data['pre_mask'].view((-1, 1))
        fut_mask = data['fut_mask'].view((-1, 1))
        pre_list = pre_mask
        fut_list = fut_mask
        cross_mask = mask.repeat((self.future_frames, self.past_frames))
        return ((fut_list).mm(pre_list.T) == 0) | cross_mask.bool()

    def futdec_self_mask_generate(self, data, mask):
        #fut_mask: (T, agent_num) -> directly mean to (agent_num), then generate a (agent_num x agent_num) mask
        fut_mask = data['fut_mask'].float().mean(0).view((-1, 1))
        fut_list = fut_mask
        return ((fut_list).mm(fut_list.T) == 0) | mask.bool()
    
    def futdec_cross_mask_generate(self, data, mask):
        #fut_mask: (T, agent_num) -> directly mean to (agent_num)
        fut_mask = data['fut_mask'].float().mean(0).view(-1, 1)
        fut_list = fut_mask
        # pre_mask: (T, agent_num) -> directly to (Txagentnum, 1)
        pre_mask = data['pre_mask'].view((-1, 1))
        pre_list = pre_mask
        mask_fut = mask.repeat((1, self.past_frames))
        return ((fut_list).mm(pre_list.T) == 0) | mask_fut.bool()

    def padding_processing(self, data, max_agent):
        N_fut, T_fut = data['fut_mask'].shape
        pad_fut_mask = torch.zeros(max_agent-N_fut, T_fut)
        data['fut_mask'] = torch.cat([data['fut_mask'], pad_fut_mask], dim=0).transpose(0, 1).contiguous()
        N_pre, T_pre = data['pre_mask'].shape
        pad_pre_mask = torch.zeros(max_agent-N_pre, T_pre)
        data['pre_mask'] = torch.cat([data['pre_mask'], pad_pre_mask], dim=0).transpose(0, 1).contiguous()
        N, T, C = data['fut_motion_orig'].shape
        pad = torch.zeros(max_agent - N, T, C)
        data['fut_motion_orig'] = torch.cat([data['fut_motion_orig'], pad],dim=0)
        data['fut_motion_orig_scene_norm'] = torch.cat([data['fut_motion_orig_scene_norm'], pad],dim=0)
        
        one_hot_state_padding = torch.zeros(max_agent-N_pre, self.state_num)
        one_hot_class_padding = torch.zeros(max_agent-N_pre, self.class_num)
        
        data['state'] = torch.cat([data['state'], one_hot_state_padding], dim=0)
        data['class'] = torch.cat([data['class'], one_hot_class_padding], dim=0)
        
        dim1_padding_keys = ['pre_motion','fut_motion',\
            'pre_motion_scene_norm','fut_motion_scene_norm',\
            'pre_vel','fut_vel','cur_motion',\
            'pre_motion_norm','fut_motion_norm']
        for k in dim1_padding_keys:
            T, N, C = data[k].shape
            pad = torch.zeros(T, max_agent-N, C)
            data[k] = torch.cat([data[k], pad],dim=1)

        if 'heading_vec' in data:
            padding_heading = torch.zeros(max_agent-data['heading_vec'].shape[0],\
                                                 data['heading_vec'].shape[1])
            data['heading_vec'] = torch.cat([data['heading_vec'], padding_heading], dim=0)

        if self.use_map:
            N_agent, C_map, H, W = data['agent_maps'].shape
            map_padding = torch.zeros(max_agent-N_agent, C_map, H, W)
            data['agent_maps'] = torch.cat([data['agent_maps'], map_padding], dim=0)

        cur_motion = data['cur_motion'][0]
        conn_dist = self.parser.get('conn_dist', 100000.0)
        
        if conn_dist < 1000.0:
            threshold = conn_dist / self.traj_scale
            pdist = F.pdist(cur_motion)
            D = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]])
            D[np.triu_indices(cur_motion.shape[0], 1)] = pdist
            D += D.T
            mask = (D>threshold)
        else:
            mask = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]])

        pre_mask = self.pre_self_mask_generate(data, mask)
        data['pastEn_self_padding_mask'] = pre_mask

        fut_self_mask = self.futenc_self_mask_generate(data, mask)
        data['futEn_self_padding_mask'] = fut_self_mask

        fut_enc_cross_mask = self.futenc_cross_mask_generate(data, mask)
        data['futEn_cross_padding_mask'] = fut_enc_cross_mask

        fut_de_self_mask = self.futdec_self_mask_generate(data, mask)
        data['futDe_self_padding_mask'] = fut_de_self_mask

        fut_de_cross_mask = self.futdec_cross_mask_generate(data, mask)
        data['futDe_cross_padding_mask'] = fut_de_cross_mask

            
        data['agent_mask'] = mask
       
        pre_input = self.pre_input_generate(data)
        data['pre_input'] = pre_input

        future_input = self.future_input_generate(data)
        data['future_input'] = future_input

        cur_input = self.cur_input_generate(data)
        data["cur_input"] = cur_input


    def __call__(self, frame):

        assert frame - self.init_frame >= 0 and frame - self.init_frame <= self.TotalFrame() - 1, 'frame is %d, total is %d' % (frame, self.TotalFrame())

        data = {}
        pre_data = self.PreData(frame)
        fut_data = self.FutureData(frame)
        valid_id = self.get_valid_id(pre_data, fut_data)
        if len(pre_data[0]) == 0 or len(fut_data[0]) == 0 or len(valid_id) == 0:
            # print(len(pre_data[0]), len(fut_data[0]), len(valid_id))
            return None

        if self.dataset == 'nuscenes_pred':
            pred_mask = self.get_pred_mask(pre_data[0], valid_id)
            heading = self.get_heading(pre_data[0], valid_id)
            one_hot_state_pre = torch.from_numpy(self.get_one_hot_state(pre_data[0], valid_id))
            one_hot_state_fut = torch.from_numpy(self.get_one_hot_state(fut_data[0], valid_id))
            one_hot_class = torch.from_numpy(self.get_one_hot_class(pre_data[0], valid_id))
        else:
            pred_mask = None
            heading = None
        
        
        if self.phase != 'training':
            data['pred_mask'] = pred_mask
            data['fut_data'] = fut_data
            data['seq'] = self.seq_name
            data['frame'] = frame
        
        data['valid_id'] = valid_id.copy()
        if len(valid_id) < self.max_agent_num and self.phase=='training':
            for _ in range(self.max_agent_num-len(valid_id)):
                data['valid_id'].append(-1)

        # pre_motion_3D: a list with [[T, 2] x agent_num] T = 4.
        # pre_motion_mask: a list with [T x agent_num], mask 0 when the agent does not move.
        
        pre_motion_3D, pre_motion_mask = self.PreMotion(pre_data, valid_id)
        fut_motion_3D, fut_motion_mask = self.FutureMotion(fut_data, valid_id)
        
        data['agent_num'] = len(pre_motion_3D)
        

        # pre_motion_stack: list -> agent_num, T, 2 -> T, agent_num, 2
        pre_motion_stack = torch.stack(pre_motion_3D, dim=0).transpose(0, 1).contiguous() #AgentN, T, 2 -> T, AgentN, 2
        fut_motion_stack = torch.stack(fut_motion_3D, dim=0).transpose(0, 1).contiguous()
        fut_motion_orig = torch.stack(fut_motion_3D, dim=0) #AgentN, T, 2
        fut_mask = torch.stack(fut_motion_mask, dim=0)
        pre_mask = torch.stack(pre_motion_mask, dim=0)
        scene_orig = pre_motion_stack[-1].mean(dim=0) #(1, 2) shape
        data['scene_orig'] = scene_orig.clone() # edited by TL, to add back on after prediction

        if heading is not None:
            # each agent has a heading.
            heading = torch.tensor(heading).float()
        
        # rotate the scene
        if self.rand_rot_scene and self.phase == "training":
            if self.discrete_rot:
                theta = torch.randint(high=24, size=(1,)) * (np.pi / 12)
            else:
                theta = torch.rand(1) * np.pi * 2
            
            pre_motion_stack, pre_motion_stack_scene_norm = self.rotation_2d_torch(pre_motion_stack, theta, scene_orig)
            fut_motion_stack, fut_motion_stack_scene_norm = self.rotation_2d_torch(fut_motion_stack, theta, scene_orig)
            fut_motion_orig, fut_motion_orig_scene_norm = self.rotation_2d_torch(fut_motion_orig, theta, scene_orig)

            if heading is not None:
                heading += theta
        else:
            theta = torch.zeros(1)
            pre_motion_stack_scene_norm = pre_motion_stack - scene_orig
            fut_motion_stack_scene_norm = fut_motion_stack - scene_orig
            fut_motion_orig_scene_norm = fut_motion_orig - scene_orig  

        pre_vel = pre_motion_stack[1:] - pre_motion_stack[:-1,:]
        fut_vel = fut_motion_stack - torch.cat([pre_motion_stack[[-1]], fut_motion_stack[:-1, :]])
        cur_motion = pre_motion_stack[[-1]]
        pre_motion_norm = pre_motion_stack[:-1] - cur_motion
        fut_motion_norm = fut_motion_stack-cur_motion

        # todo dynamic padding
        # if self.dynamic_padding:
        data['pre_motion'] = pre_motion_stack
        data['fut_motion'] = fut_motion_stack
        data['fut_motion_orig'] = fut_motion_orig
        data['fut_mask'] = fut_mask 
        data['pre_mask'] = pre_mask
        data['pre_motion_scene_norm'] = pre_motion_stack_scene_norm
        data['fut_motion_scene_norm'] = fut_motion_stack_scene_norm
        data['fut_motion_orig_scene_norm'] = fut_motion_orig_scene_norm
        data['pre_vel'] = pre_vel
        data['fut_vel'] = fut_vel
        data['cur_motion'] = cur_motion
        data['pre_motion_norm'] = pre_motion_norm
        data['fut_motion_norm'] = fut_motion_norm
        data['state'] = one_hot_state_pre
        data['class'] = one_hot_class
        
        if heading is not None:
            data['heading_vec'] = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)

        if self.use_map:
            scene_map = self.geom_scene_map
            scene_points = np.stack(pre_motion_3D)[:, -1] * self.traj_scale
            if self.map_global_rot:
                patch_size = [50, 50, 50, 50]
                rot = theta.repeat(agent_num).cpu().numpy() * (180 / np.pi)
            else:
                patch_size = [50, 10, 50, 90]
                rot = -np.array(heading)  * (180 / np.pi)
            data['agent_maps'] = scene_map.get_cropped_maps(scene_points, patch_size, rot)
           
            if self.map_moco and self.phase == 'training':
                data['agent_maps_moco'] = data['agent_maps'].clone()
                whole_map = scene_map.data
                C, h, w = whole_map.shape
                ps1, ps2 = data['agent_maps'][0].shape[1:]
                x_list, y_list = whole_map.mean(axis=0).nonzero()
                
                index_list = (x_list>ps1//2) & (x_list<h-ps1//2) & (y_list>ps2//2) & (y_list<w-ps2//2)
                x_list = x_list[index_list]
                y_list = y_list[index_list]
                whole_map = torch.from_numpy(whole_map)
                random_list = np.random.choice(range(len(x_list)), self.max_agent_num - data['agent_num'] + self.moco_num)
                center_point_x = x_list[random_list]
                center_point_y = y_list[random_list]
                moco_list = []
                for i in range(len(center_point_x)):
                    x = center_point_x[i]
                    y = center_point_y[i]
                    moco_list.append(whole_map[:, x-ps1//2:x+ps1//2, y-ps2//2:y+ps2//2])
                # cv2.imwrite('./eax.png', np.transpose(moco_list[0].numpy(), (2, 1, 0)).astype(np.uint))
                moco_list = torch.stack(moco_list)
                data['agent_maps_moco'] = torch.cat([data['agent_maps_moco'], moco_list], dim=0)

        if not self.dynamic_padding:
            self.padding_processing(data, self.max_agent_num)
        else:
            data['preprocessor'] = self

        return data
