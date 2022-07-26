from .nuscenes_pred_split import get_nuscenes_pred_split
import os, random, numpy as np, copy
import os

from .preprocessor import preprocess
import torch
from torch.utils.data import DataLoader

def print_log(print_str, log, same_line=False, display=True):
	'''
	print a string to a log file

	parameters:
		print_str:          a string to print
		log:                a opened file to save the log
		same_line:          True if we want to print the string without a new next line
		display:            False if we want to disable to print the string onto the terminal
	'''
	if display:
		if same_line: print('{}'.format(print_str), end='')
		else: print('{}'.format(print_str))

	if same_line: log.write('{}'.format(print_str))
	else: log.write('{}\n'.format(print_str))
	log.flush()

class nuscenes_dataloader(torch.utils.data.Dataset):

    def __init__(self, parser, log, split='train', phase='training'):
        self.past_frames = parser.past_frames
        self.min_past_frames = parser.min_past_frames
        self.frame_skip = parser.get('frame_skip', 1)
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'
        data_root = parser.data_root_nuscenes_pred           
        seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
        self.init_frame = 0
        process_func = preprocess
        self.data_root = data_root
        
     
        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test': self.sequence_to_load = seq_test
        else:                      assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        for seq_name in self.sequence_to_load:
            preprocessor = process_func(data_root, seq_name, parser, log, self.split, self.phase)
            num_seq_samples = preprocessor.num_fr - (parser.min_past_frames - 1) * self.frame_skip - parser.min_future_frames * self.frame_skip + 1
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)

        self.sample_list = list(range(self.num_total_samples))
        self.index = 0

        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)


    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        sample_index = self.sample_list[index]
        seq_index, frame = self.get_seq_and_frame(sample_index)
        seq = self.sequence[seq_index]
        data = seq(frame)
        if data is None:
            if (index+1) >= len(self.sample_list):
                return self.__getitem__(np.random.choice(range(len(self.sample_list))))
            else:
                return self.__getitem__(index+1)
        else:
            return data

dataset_factory = {
    "nuscenes_pred": nuscenes_dataloader,
}

if __name__ == '__main__':
    import os
    import sys
    import argparse
    import numpy as np
    import torch
    sys.path.append(os.getcwd())
    from utils.torch import *
    from utils.config import Config
    from utils.utils import prepare_seed, print_log, AverageMeter, convert_secs2time, get_timestring
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='k10_res')
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--dataset', default="nuscenes_pred")
    parser.add_argument('--batch_size', default=11)
    parser.add_argument('--num_workers', default=16)
    args = parser.parse_args()
    seed = 1
    cfg = Config(args.cfg, args.tmp, create_dirs=True)
    split="train"
    log = open(os.path.join(cfg.log_dir, 'log.txt'), 'a+')
    dataset = dataset_factory[args.dataset](cfg, log, split=split, phase='training')

    loaders = DataLoader(
            dataset, shuffle=(split == 'train'), batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
        )

    for idx, data in enumerate(loaders):
        print(idx)
        print(data.keys())
        # print(data['agent_mask'])
        for k,v in data.items():
            print(k,v.shape)
        break
