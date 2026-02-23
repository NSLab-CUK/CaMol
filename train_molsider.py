
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
loss_func = nn.CrossEntropyLoss()
from metrics import accuracy_TU, MAE
 

def train_epoch(model, args, optimizer, device, data_loader, epoch, k_transition, batch_size = 16 ):

	model.train()

	epoch_loss = 0; epoch_KL_Loss = 0; epoch_contrastive_loss = 0 ;  epoch_reconstruction_loss = 0
	nb_data = 0
	gpu_mem = 0
	count = 0
	for iter, (batch_graphs, _ , batch_subgraphs, batch_logMs) in enumerate(data_loader):
 
		count =iter
		batch_graphs = batch_graphs.to(device)
		batch_x = batch_graphs.ndata['x'].float().to(device) 
		edge_index = batch_graphs.edges()
		
		optimizer.zero_grad()
		flatten_batch_subgraphs = list(chain.from_iterable(batch_subgraphs))
		flatten_batch_subgraphs  = dgl.batch(flatten_batch_subgraphs).to(device) 
		x_subs = flatten_batch_subgraphs.ndata['x'].float().to(device) 

		_, KL_Loss, contrastive_loss, reconstruction_loss = model.forward(batch_graphs,batch_x,flatten_batch_subgraphs , batch_logMs, x_subs,  1, edge_index, 2, device, batch_size)
		parameters = []
		for parameter in model.parameters():
			parameters.append(parameter.view(-1))
		loss = KL_Loss + reconstruction_loss + contrastive_loss
		loss.backward()
		optimizer.step()
		epoch_loss += loss.detach().item()
		epoch_KL_Loss+= KL_Loss;        epoch_contrastive_loss+= contrastive_loss;      epoch_reconstruction_loss+= reconstruction_loss

	epoch_loss /= (count + 1)
	epoch_KL_Loss/= (count + 1);     epoch_contrastive_loss/= (count + 1);            epoch_reconstruction_loss/= (count + 1)
	return epoch_loss, epoch_KL_Loss, epoch_contrastive_loss, epoch_reconstruction_loss

from ogb.graphproppred import Evaluator
def process_diff(batch_adj, batch_size):
	list_batch = []
	max_size = 0
	for i in range(batch_size):
		size= batch_adj[i].size(dim=1)
		if size> max_size:
			max_size = size
	
	p2d = (0,2,0,2) 
	for i in range(batch_size):
		diff= max_size - batch_adj[i].size(dim=1)
		if diff != max_size:
			p2d = (0,diff,0,diff) # pad last dim by 1 on each side
			batch_adj[i] = F.pad(batch_adj[i], p2d, "constant", 0) 

			list_batch.append(batch_adj[i])
	return torch.stack(batch_adj, dim=0)
	#raise SystemExit()
import dgl
from MetricWrapper import MetricWrapper
from itertools import chain
from torch import optim
import torch.nn.functional as F
import copy

def maml_inner_loop(args, list_label_support, fast_model, task_id, batch_graph_support, batch_support_x,
					batch_graph_query, batch_query_x,
					batch_context, batch_context_x,edge_context_feature,
					device, batch_size, inner_lr,
					flatten_batch_subgraphs_s,x_subs_s,flatten_batch_subgraphs_q,x_subs_q):

	sup_pre, causal_pre, int_pre, random_pre = fast_model.forward(task_id, batch_graph_support, batch_support_x,
					batch_graph_query, batch_query_x,
					batch_context, batch_context_x,edge_context_feature,
					device, batch_size, flatten_batch_subgraphs_s,x_subs_s,flatten_batch_subgraphs_q,x_subs_q , 'inner')

	list_label_support = list_label_support.squeeze(0).to(device)
	sup_loss = fast_model.loss(sup_pre, list_label_support)
	
	causal_loss = fast_model.loss(causal_pre, list_label_support); 
	int_loss = fast_model.loss(int_pre, list_label_support)

	y_rand = (torch.ones_like(random_pre ) / 2).to(device)
	rand_loss = fast_model.loss(random_pre, y_rand)

	loss = ( sup_loss +causal_loss + int_loss+ rand_loss)/ 4

	grads = torch.autograd.grad(loss, fast_model.parameters(), create_graph=True, allow_unused=True)
 
	fast_weights = []
	for (name, param), grad in zip(fast_model.named_parameters(), grads):
		fw = param - inner_lr * grad
 
	return fast_weights

def train_epoch_graph_classification(args, model, meta_optimizer, device, data_loader, epoch, batch_size):
	
	epoch_train_auc = 0
	targets = torch.tensor([]).to(device)
	scores = torch.tensor([]).to(device)
	meta_optimizer.zero_grad()
	model.train()

	#each task:
	tasks_per_epoch = 0; total_meta_loss = 0
	for iter, (list_graph_support,list_label_support,list_graph_query,list_label_query,context_graph, task_id, 
			list_subgraph_support, list_subgraph_query) in enumerate(data_loader):
		tasks_per_epoch+=1

 
		batch_graph_support = list(chain.from_iterable(list_graph_support))
		batch_graph_support = dgl.batch(batch_graph_support).to(device)
		batch_support_x = batch_graph_support.ndata['x_2'].float().to(device)
 
		batch_graph_query = list(chain.from_iterable(list_graph_query))
		batch_graph_query = dgl.batch(batch_graph_query).to(device)
		batch_query_x = batch_graph_query.ndata['x_2'].float().to(device)
 
		batch_context = context_graph.to(device)
		batch_context_x = batch_context.ndata['x'].float().to(device)
		edge_context_feature = batch_context.edata['e'] #; edge_feature = edge_feature.unsqueeze(1).float()
 
 
		flatten_batch_subgraphs_s = list(chain.from_iterable(list_subgraph_support)); flatten_batch_subgraphs_q= list(chain.from_iterable(list_subgraph_query))
		flatten_batch_subgraphs_s = list(chain.from_iterable(flatten_batch_subgraphs_s)); flatten_batch_subgraphs_q= list(chain.from_iterable(flatten_batch_subgraphs_q))
		 
		flatten_batch_subgraphs_s  = dgl.batch(flatten_batch_subgraphs_s).to(device)  
		x_subs_s = flatten_batch_subgraphs_s.ndata['x_2'].float().to(device); 
	
		flatten_batch_subgraphs_q  = dgl.batch(flatten_batch_subgraphs_q).to(device)  
		x_subs_q = flatten_batch_subgraphs_q.ndata['x_2'].float().to(device); 
		x_subs_s  = F.normalize(x_subs_s) ; x_subs_q  = F.normalize(x_subs_q)
  
		# Normalization
		if args.norm == 1:
			batch_support_x  = F.normalize(batch_support_x); batch_query_x= F.normalize(batch_query_x)
			batch_context_x  = F.normalize(batch_context_x) 
 
		fast_weights  = maml_inner_loop(args, list_label_support, model, task_id, batch_graph_support, batch_support_x,
																		batch_graph_query, batch_query_x,
																		batch_context, batch_context_x,edge_context_feature,
																		device, batch_size, args.lr_inner,
																		flatten_batch_subgraphs_s,x_subs_s,
																		flatten_batch_subgraphs_q,x_subs_q)
		idx = 0
		for name, param in model.named_parameters():
			param_tmp = fast_weights[idx]
			param.data = param_tmp.data.clone()
			idx += 1

		query_pre, causal_pre, int_pre, random_pre  = model.forward(task_id, batch_graph_support, batch_support_x,
								batch_graph_query, batch_query_x,
								batch_context, batch_context_x,edge_context_feature,
								device, batch_size, flatten_batch_subgraphs_s,x_subs_s,flatten_batch_subgraphs_q,x_subs_q, 'outer')
		
		list_label_query = list_label_query.squeeze(0).to(device)

		query_loss = model.loss(query_pre, list_label_query)
		causal_loss = model.loss(causal_pre, list_label_query)
		int_loss = model.loss(int_pre, list_label_query)

		y_rand = (torch.ones_like(random_pre ) / 2).to(device)
		rand_loss = model.loss(random_pre, y_rand)
 
		tot_loss = (query_loss + causal_loss +int_loss + rand_loss) /4

		total_meta_loss+= tot_loss
 
		targets = torch.cat((targets, list_label_query), 0) ; scores = torch.cat((scores, query_pre), 0)
 
		if tasks_per_epoch % args.tasks_per_epoch == 0:
			break
	avg_meta_loss = total_meta_loss / args.tasks_per_epoch
	avg_meta_loss.backward()
	meta_optimizer.step()
 
	y_score = scores.detach().cpu().numpy(); y_true = targets.detach().cpu().numpy()
	epoch_train_auc = roc_auc_score(y_true, y_score)
	return avg_meta_loss.detach().item(), epoch_train_auc, meta_optimizer

from sklearn.metrics import roc_auc_score
def evaluate_network(args, model, meta_optimizer, device, data_loader, epoch, batch_size):
	# model.eval()
	epoch_train_auc = 0
	targets = torch.tensor([]).to(device); scores = torch.tensor([]).to(device)


	for iter, (list_graph_support,list_label_support,list_graph_query,list_label_query,context_graph, task_id, 
			list_subgraph_support, list_subgraph_query) in enumerate(data_loader):

		#support 
		batch_graph_support = list(chain.from_iterable(list_graph_support))
		batch_graph_support = dgl.batch(batch_graph_support).to(device)
		batch_support_x = batch_graph_support.ndata['x_2'].float().to(device)
		#query
		batch_graph_query = list(chain.from_iterable(list_graph_query))
		batch_graph_query = dgl.batch(batch_graph_query).to(device)
		batch_query_x = batch_graph_query.ndata['x_2'].float().to(device)
		#context
		batch_context = context_graph.to(device)
		batch_context_x = batch_context.ndata['x'].float().to(device)
		edge_context_feature = batch_context.edata['e'] #; edge_feature = edge_feature.unsqueeze(1).float()

		#s-cgib
		flatten_batch_subgraphs_s = list(chain.from_iterable(list_subgraph_support)); flatten_batch_subgraphs_q= list(chain.from_iterable(list_subgraph_query))
		flatten_batch_subgraphs_s = list(chain.from_iterable(flatten_batch_subgraphs_s)); flatten_batch_subgraphs_q= list(chain.from_iterable(flatten_batch_subgraphs_q))

		flatten_batch_subgraphs_s  = dgl.batch(flatten_batch_subgraphs_s).to(device)  
		x_subs_s = flatten_batch_subgraphs_s.ndata['x_2'].float().to(device); 
	
		flatten_batch_subgraphs_q  = dgl.batch(flatten_batch_subgraphs_q).to(device)  
		x_subs_q = flatten_batch_subgraphs_q.ndata['x_2'].float().to(device); 
		x_subs_s  = F.normalize(x_subs_s) ; x_subs_q  = F.normalize(x_subs_q)

		# Normalization
		if args.norm ==1:
			batch_support_x  = F.normalize(batch_support_x); batch_query_x= F.normalize(batch_query_x); 
			batch_context_x  = F.normalize(batch_context_x); 

		# Inner loop
		fast_weights = maml_inner_loop(args, list_label_support, model, task_id, batch_graph_support, batch_support_x,
																			batch_graph_query, batch_query_x,
																			batch_context, batch_context_x,edge_context_feature,
																			device, batch_size, args.lr_inner,
																			flatten_batch_subgraphs_s,x_subs_s,
																			flatten_batch_subgraphs_q,x_subs_q)
		idx = 0
		for name, param in model.named_parameters():
			param_tmp = fast_weights[idx]
			param.data = param_tmp.data.clone()
			idx += 1
		# Meta-update via query set
		model.eval()
		with torch.no_grad():
			_, causal_pre, _ , _ = model.forward(task_id, batch_graph_support, batch_support_x,
									batch_graph_query, batch_query_x,
									batch_context, batch_context_x, edge_context_feature,
									device, batch_size, flatten_batch_subgraphs_s,x_subs_s,flatten_batch_subgraphs_q,x_subs_q, 'outer')

			list_label_query= list_label_query.squeeze(0).to(device)

			targets = torch.cat((targets, list_label_query), 0)
			scores = torch.cat((scores, causal_pre), 0)

	y_score = scores.detach().cpu().numpy() ; 	y_true = targets.detach().cpu().numpy()
	epoch_train_auc = roc_auc_score(y_true, y_score)
	return None, epoch_train_auc