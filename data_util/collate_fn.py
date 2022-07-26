import torch

def DynamicPaddingCollate(data_list):
	max_agent = 0
	bs = len(data_list)
	for i in range(bs):
		max_agent = max(max_agent, data_list[i]['pre_motion'].shape[1])
		
	for i in range(bs):
		data_list[i]['preprocessor'].padding_processing(data_list[i], max_agent)
		data_list[i].pop('preprocessor')

	batch_data = {}
	for k in data_list[0]:
		if type(data_list[0][k]) is torch.Tensor:
			data_k = [data[k] for data in data_list]
			batch_data[k] = torch.stack(data_k,dim=0)
		elif type(data_list[0][k]) is int:
			data_k = [data[k] for data in data_list]
			batch_data[k] = torch.tensor(data_k)
		else:
			data_k = [data[k] for data in data_list]
			batch_data[k] = data_k
	return batch_data

collate_factory = {
	'dynamic_padding': DynamicPaddingCollate,
	'default': None
}