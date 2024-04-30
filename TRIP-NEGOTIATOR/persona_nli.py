from load_bert import bert_model
import torch
from torch import nn

#{0:'contradiction', 1:'entailment', 2:'neutral'}

bert = bert_model()
softmax = nn.Softmax(dim=-1)

def get_nli(uttr, persona):
	labels, logits = bert.predict_label([uttr]*len(persona), persona) 
	logits = torch.tensor(logits)
	logits = softmax(logits)
	return logits, labels


