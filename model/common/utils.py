import torch
def extract_and_aggregate_context(context_dict, context_encoders,
		seq_len):
	context_features = [context_encoders[name](context) \
						for name, context in context_dict.items()]
	for i in range(len(context_features)):
		# if the output is not sequential data, repeat it.
		# print(context_features[i].shape, 'what is the single shape')
		if context_features[i].ndim == 2:
			context_features[i] = context_features[i].unsqueeze(0)\
						.repeat(seq_len,1,1) # seq, batch, channel
	return context_features