import argparse
import copy
import logging
import math
import time
from pathlib import Path
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
import os
import random
import dgl
from util import getM_logM, load_dgl, get_A_D, load_dgl_benzene, get_mol, motif_decomp, get_3D

from script_classification import run_node_classification, run_node_clustering, update_evaluation_value
from train_molsider import train_epoch, evaluate_network, train_epoch_graph_classification
from torch.utils.data import DataLoader
import gzip, pickle
import collections
from collections import defaultdict
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import TUDataset, ZINC, MoleculeNet
from dgl.data.utils import save_graphs, load_graphs
from models import Meta_model
from rdkit import Chem, RDLogger
import warnings


from gnnutils import make_masks, train, test, add_original_graph, load_webkb, load_planetoid, load_wiki, load_bgp, \
	load_film, load_airports, load_amazon, load_coauthor, load_WikiCS, load_crocodile, load_Cora_ML
from util import get_B_sim_phi, getM_logM, load_dgl, get_A_D, load_dgl_fromPyG


warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='scipy._lib.messagestream.MessageStream')
RDLogger.DisableLog('rdApp.*')
np.seterr(divide='ignore')


def collate(self, samples):
	graphs, labels = map(list, zip(*samples))
	labels = torch.tensor(np.array(labels)).unsqueeze(1)
	batched_graph = dgl.batch(graphs)

	return batched_graph, labels

def count_params(module):
    return sum(p.numel() for p in module.parameters()  )
def run(i, dataset_full, num_features, num_classes):
	model = Meta_model(args).to(device)
	# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
	best_model = copy.deepcopy(model)
	trainset, valset, testset = dataset_full.train, dataset_full.val, dataset_full.test
	data_all = dataset_full.data_all
	print("\nTraining Graphs: ", len(trainset))
	print("Validation Graphs: ", len(valset))
	print("Test Graphs: ", len(testset))

	batch_size = args.batch_size

	train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=dataset_full.collate)
	val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=dataset_full.collate)
	test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=dataset_full.collate)
	# pre_train_loader = DataLoader(data_all, batch_size= batch_size, shuffle=False, collate_fn = dataset_full.collate)

	runs_acc = []
	for i in tqdm(range(args.run_times)):
		print(f'\nrun time: {i}')
		acc, best_epoch = run_epoch_graph_classification(best_model, train_loader, val_loader, test_loader,
														 num_features, batch_size=batch_size)
		runs_acc.append(acc)
		time.sleep(0.1)

	runs_acc = np.array(runs_acc) * 100
	final_msg = "Graph classification: Mean %0.4f, Std %0.4f" % (runs_acc.mean(), runs_acc.std())
	print(final_msg)


def run_epoch_graph_classification(model, train_loader, val_loader, test_loader, num_features, batch_size):
	best_model = model
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_inner, weight_decay=1e-5)
	best_loss = 100000000
	t0 = time.time()
	per_epoch_time = []
	epoch_train_AUCs, epoch_val_AUCs, epoch_test_AUCs = [], [], []
	epoch_train_losses, epoch_val_losses = [], []
	for epoch in range(1, args.fn_epoches):
		start = time.time()
		epoch_train_loss, epoch_train_auc, optimizer = train_epoch_graph_classification(args, model, optimizer, device,
																						train_loader, epoch, batch_size)

		_, epoch_test_auc = evaluate_network(args, model, optimizer, device, test_loader, epoch, batch_size)
		epoch_train_losses.append(epoch_train_loss)
		# epoch_val_losses.append(epoch_val_loss)
		epoch_train_AUCs.append(epoch_train_auc)
		# epoch_val_AUCs.append(epoch_val_auc)
		epoch_test_AUCs.append(epoch_test_auc)
		if best_loss >= epoch_train_loss:
			best_model = model; best_epoch = epoch; best_loss = epoch_train_loss
		if epoch - best_epoch > 500:
			print(f"Finish epoch - best_epoch > 100")
			break
		if epoch % 1 == 0:
			print(
				f'Epoch: {epoch}	|Best_epoch: {best_epoch}	|Train_loss: {np.round(epoch_train_loss, 6)} 	| Train_auc: {np.round(epoch_train_auc, 6)}	| epoch_test_auc: {np.round(epoch_test_auc, 6)} ')
		per_epoch_time.append(time.time() - start)
		# Stop training after params['max_time'] hours
		if time.time() - t0 > 48 * 3600:
			print('-' * 89)
			print("Max_time for training elapsed")
			break
	_, test_acc = evaluate_network(args, best_model, optimizer, device, test_loader, epoch, batch_size)  # (best_model, device, test_loader, epoch)
	print("Convergence Time (Epochs): {:.4f}".format(epoch))
	print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
	print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

	return test_acc, best_epoch

def main():
	timestr = time.strftime("%Y%m%d-%H%M%S")
	log_file = args.dataset + "-" + timestr + ".log"
	Path("./exp_logs").mkdir(parents=True, exist_ok=True)
	logging.basicConfig(filename="exp_logs/" + log_file, filemode="w", level=logging.INFO)
	logging.info("Starting on device: %s", device)
	logging.info("Config: %s ", args)
	if args.dataset in [ "MUV",  "SIDER",  "Tox21", "ToxCast", "ClinTox"]:
		dataset = MoleculeNet(root='original_datasets/' + args.dataset, name = args.dataset)
		args.num_features_org = dataset.num_features
	else:
		raise NotImplementedError
	print(f'Checking dataset {args.dataset}')
	if args.dataset in ["ToxCast"]:
		if args.sub_dataset == 'ToxCast-APR':
			args.num_tasks = 43;args.num_test_p = 10
		if args.sub_dataset == 'ToxCast-ATG':
			args.num_tasks = 146;args.num_test_p = 40
		if args.sub_dataset == 'ToxCast-BSK':
			args.num_tasks = 115;args.num_test_p = 31
		###
		if args.sub_dataset == 'ToxCast-CEETOX':
			args.num_tasks = 14;args.num_test_p = 4
		if args.sub_dataset == 'ToxCast-CLD':
			args.num_tasks = 19;args.num_test_p = 5
		if args.sub_dataset == 'ToxCast-NVS':
			args.num_tasks = 139;args.num_test_p = 39
		###
		if args.sub_dataset == 'ToxCast-OT':
			args.num_tasks = 15;args.num_test_p = 4
		if args.sub_dataset == 'ToxCast-TOX21':
			args.num_tasks = 100;args.num_test_p = 20
		if args.sub_dataset == 'ToxCast-Tanguay':
			args.num_tasks = 18;args.num_test_p = 5
	elif args.dataset in ["Tox21"]:
		args.num_tasks = 12;args.num_test_p = 3
	elif args.dataset in ["SIDER"]:
		args.num_tasks = 27;args.num_test_p = 6
	elif args.dataset in ["MUV"]:
		args.num_tasks = 17;args.num_test_p = 5

	fi = f"pts/{args.dataset}_{args.num_eposides}_kshot_{args.k_shot}_{args.num_query}_{args.sub_dataset}.pkl.gz"
	if not os.path.exists(fi):
		print(f"generating samples {fi}...")
		dataset_full = generate_graphs_new_fs(fi,dataset)
	else:
		print(f"loading samples from {fi}...")
		samples_all = torch.load(fi)
		dataset_full = LoadData(samples_all, args.dataset)


	run(0, dataset_full, args.num_features, args.num_classes)

	raise SystemExit()


import pandas as pd


def sampling_indices_SQ(list_smiles, list_p_tensor, task_id):
	task_values = list_p_tensor[:, task_id]

	matching_indices = (task_values == 1).nonzero(as_tuple=True)[0]
	if len(matching_indices) < args.k_shot or len(matching_indices) < args.num_query / 2:
		raise ValueError("Not enough matching elements to sample args.k_shot.")
	s_1_indices = random.sample(matching_indices.tolist(), args.k_shot)
	q_1_indices = random.sample(matching_indices.tolist(), int(args.num_query / 2))


	matching_indices = (task_values == 0).nonzero(as_tuple=True)[0]
	if len(matching_indices) < args.k_shot or len(matching_indices) < args.num_query / 2:
		raise ValueError("Not enough matching elements to sample args.k_shot.")
	s_0_indices = random.sample(matching_indices.tolist(), args.k_shot)
	q_0_indices = random.sample(matching_indices.tolist(), int(args.num_query / 2))

	s_indices = s_1_indices + s_0_indices
	q_indices = q_1_indices + q_0_indices

	return s_indices, q_indices


def indices2graph_list(s_indices, q_indices, graph_lists, graph_labels, list_p_tensor, task_id, dataset):
	task_values = list_p_tensor[:, task_id]
	list_graph_support = [];
	list_label_support = []
	list_graph_query = [];
	list_label_query = []
	list_graph_nots = [];
	list_label_nots = []
	# support set
	num_s = len(s_indices)
	list_subgraph_support = []; list_subgraph_query = []
	for mol_id in range(num_s):
		current_s_id = s_indices[mol_id]
		# graph
		current_graph_s = graph_lists[current_s_id]
		#s-cgib
		data = dataset[current_s_id]
		current_graph_s.ndata['x_2'] = data.x

		list_graph_support.append(current_graph_s)

		current_property_s = task_values[current_s_id]
		list_label_support.append(current_property_s)
		
		#s-cgib subgraphs
		node_ids = current_graph_s.nodes()
		all_subgraphs = [dgl.khop_in_subgraph(current_graph_s, individual_node, k= args.k_transition)[0] for individual_node in node_ids]
		list_subgraph_support.append(all_subgraphs)

	# query set
	num_q = len(q_indices)
	for mol_id in range(num_q):
		current_q_id = q_indices[mol_id]
		# graph
		current_graph_q = graph_lists[current_q_id]
		#s-cgib
		data = dataset[current_q_id]
		current_graph_q.ndata['x_2'] = data.x
		# 
		list_graph_query.append(current_graph_q)

		current_property_q = task_values[current_q_id]
		list_label_query.append(current_property_q)
		#s-cgib subgraphs
		node_ids = current_graph_q.nodes()
		all_subgraphs = [dgl.khop_in_subgraph(current_graph_q, individual_node, k= args.k_transition)[0] for individual_node in node_ids]
		list_subgraph_query.append(all_subgraphs)


	list_label_support = torch.stack(list_label_support).unsqueeze(1)
	list_label_query = torch.stack(list_label_query).unsqueeze(1)

	return list_graph_support, list_label_support, list_graph_query, list_label_query , list_subgraph_support, list_subgraph_query


def generate_graphs_new_fs(fi_sample, dataset):
	if args.dataset in ["ToxCast"]:
		#args.num_tasks = 617;         args.num_test_p = 158
		file_name = "original_datasets/ToxCast/toxcast/raw/toxcast_data.csv"
		data = pd.read_csv(file_name)
		d_smile = data['smiles']
		#d_class = data.iloc[:, 1:]
		if args.sub_dataset == 'ToxCast-APR':
			selected_columns = [col for col in data.columns if "APR" in col]
			d_class = data[selected_columns]
		if args.sub_dataset == 'ToxCast-ATG':
			selected_columns = [col for col in data.columns if "ATG" in col]
			d_class = data[selected_columns]
		if args.sub_dataset == 'ToxCast-BSK':
			selected_columns = [col for col in data.columns if "BSK" in col]
			d_class = data[selected_columns]
		#
		if args.sub_dataset == 'ToxCast-CEETOX':
			selected_columns = [col for col in data.columns if "CEETOX" in col]
			d_class = data[selected_columns]
		if args.sub_dataset == 'ToxCast-CLD':
			selected_columns = [col for col in data.columns if "CLD" in col]
			d_class = data[selected_columns]
		if args.sub_dataset == 'ToxCast-NVS':
			selected_columns = [col for col in data.columns if "NVS" in col]
			d_class = data[selected_columns]
		#
		if args.sub_dataset == 'ToxCast-OT':
			selected_columns = [col for col in data.columns if "OT" in col]
			d_class = data[selected_columns]
		if args.sub_dataset == 'ToxCast-TOX21':
			selected_columns = [col for col in data.columns if "TOX21" in col]
			d_class = data[selected_columns]
		if args.sub_dataset == 'ToxCast-Tanguay':
			selected_columns = [col for col in data.columns if "Tanguay" in col]
			d_class = data[selected_columns]
	elif args.dataset in ["ClinTox"]:
		file_name = "original_datasets/ClinTox/clintox/raw/clintox.csv"
		data = pd.read_csv(file_name)
		d_smile = data['smiles']
		d_class = data.iloc[:, 1:]
	elif args.dataset in ["Tox21"]:
		args.num_tasks = 12
		args.num_test_p = 3
		file_name = "original_datasets/Tox21/tox21/raw/tox21.csv"
		data = pd.read_csv(file_name)
		d_smile = data['smiles']
		d_class = data.iloc[:, 0: args.num_tasks]
	elif args.dataset in ["SIDER"]:
		args.num_tasks = 27
		args.num_test_p = 6
		file_name = "original_datasets/SIDER/sider/raw/sider.csv"
		data = pd.read_csv(file_name)
		d_smile = data['smiles']
		d_class = data.iloc[:, 1:28]
	elif args.dataset in ["MUV"]:
		args.num_tasks = 17
		args.num_test_p = 5
		file_name = "original_datasets/MUV/muv/raw/muv.csv"
		data = pd.read_csv(file_name)
		d_smile = data['smiles']
		d_class = data.iloc[:, :args.num_tasks]
	else:
		print(f" checking datasets ...")
		raise SystemExit()

	# Loading graphs with learned node features from S-CGIB
	fi = f"pts/{args.dataset}"
	graph_lists, graph_labels = load_graphs(fi + "_node_feature.bin")
	graph_feature = torch.load(fi + "_graph_feature.pt");
	graph_feature_tensor = torch.stack(graph_feature)
	smiles_id_filted = torch.load(fi + "_smiles_id.pt")   

	print(
		f"len(graph_lists): {len(graph_lists)}| smiles_id {len(smiles_id_filted)}")  
	print(f"graph_labels: {graph_labels['glabel'].shape} ")  
	print(f"graph_feature: {graph_feature_tensor.shape} ")   
	print(f"smiles_id: {len(smiles_id_filted)}")  

	# store smiles and properties
	list_smiles = [];
	list_p = []
	graph_lists_new= []
	new_dataset= []
	idx = 0
	for i, row in data.iterrows():
		if i not in smiles_id_filted:
			continue
		if i % 1000 == 0:
			print(f"processing mol {i}")
			time.sleep(0.1)
		smile = d_smile[i]
		y = d_class.iloc[i].to_numpy().tolist()
		list_smiles.append(smile);
		list_p.append(y)

		current_g = graph_lists[idx]
		graph_lists_new.append(current_g)
		data = dataset[i]; new_dataset.append(data)
		idx+=1

	list_p_tensor = torch.tensor(list_p)

	k_shot = args.k_shot
	num_query = args.num_query  # 5 5 10
	num_nodes_context = k_shot * 2 + num_query  # 20

	num_nodes_context = num_nodes_context + args.num_tasks  # 32

	X = torch.rand(args.num_tasks, args.num_features)
	samples_all = []
	num_training = args.num_eposides * 0.8

	for i in range(args.num_eposides):
		if i % 100 == 0:
			print(f"sampling {i} ...")

		if i < num_training:
			task_id = random.randint(0, args.num_tasks - 1)
		else:
			task_id = random.randint(args.num_tasks - args.num_test_p, args.num_tasks - 1)

		s_indices, q_indices = sampling_indices_SQ(list_smiles, list_p_tensor, task_id)
 
		list_graph_support, list_label_support, list_graph_query, list_label_query, list_subgraph_support, list_subgraph_query = indices2graph_list(
			s_indices, q_indices,  graph_lists_new, graph_labels, list_p_tensor, task_id, new_dataset)

 
		context_graph = build_context_graph(X, list_p, num_nodes_context, list_smiles, graph_feature_tensor,
											graph_labels, list_graph_support, list_label_support,
											list_graph_query, list_label_query, s_indices, q_indices)

		# 1 task --> 1 batch: put to the pair
		pair = (list_graph_support, list_label_support, list_graph_query, list_label_query, context_graph, task_id, list_subgraph_support, list_subgraph_query)
		samples_all.append(pair)
	print('Saving the dataset')
	torch.save(samples_all, fi_sample)
 
	print('Saved')
	dataset_full = LoadData(samples_all, args.dataset)
	return dataset_full


def handling_support_samples(list_graph_features, s_indices, num_s, list_graph_support, list_smiles, fg_mol_features,
							 src, global_id):
	for mol_id in range(num_s):
		current_support_id = s_indices[mol_id]
		current_graph = list_graph_support[mol_id]
		current_s_node_x = current_graph.ndata['x']  # num_nodes = current_graph.num_nodes()
		list_graph_features.append(current_s_node_x.sum(dim=0))
		node_ids = current_graph.nodes()
		smile = list_smiles[current_support_id]
		mol = get_mol(smile)
		cliques = motif_decomp(mol)
		for ind_node in node_ids:
			check_exsit_in_fgs = 0
			for cliq in cliques:
				if ind_node in cliq:
					FG_feature = current_s_node_x[cliq].sum(dim=0)

					fg_mol_features.append(FG_feature)
					src.append(global_id)
					global_id += 1
					check_exsit_in_fgs = 1
					continue

	return list_graph_features, fg_mol_features, global_id, fg_mol_features, src


def handling_query_samples(list_graph_features, q_indices, num_q, list_graph_query, list_smiles, fg_mol_features, src,
						   global_id):
	for mol_id in range(num_q):
		current_query_id = q_indices[mol_id]

		current_graph = list_graph_query[mol_id]
		current_q_node_x = current_graph.ndata['x']  # num_nodes = current_graph.num_nodes()
		list_graph_features.append(current_q_node_x.sum(dim=0))
		node_ids = current_graph.nodes()
		smile = list_smiles[current_query_id]
		mol = get_mol(smile)
		cliques = motif_decomp(mol)
		for ind_node in node_ids:
			check_exsit_in_fgs = 0
			for cliq in cliques:
				if ind_node in cliq:
					FG_feature = current_q_node_x[cliq].sum(dim=0)
					fg_mol_features.append(FG_feature)
					src.append(global_id)
					global_id += 1;
					check_exsit_in_fgs = 1
					continue

	return list_graph_features, fg_mol_features, global_id, fg_mol_features, src


def build_FG_Mols(X, list_p, num_nodes_context, list_smiles, graph_feature_tensor, graph_labels, list_graph_support,
				  list_label_support,
				  list_graph_query, list_label_query, s_indices, q_indices):
	src = [];
	dst = [];
	edge_type = []  # 5 mols + 12:1 +12:0 +12:n = 41 nodes
	k_shot = args.k_shot;
	num_query = args.num_query  # 5 5 10
	global_id = 0
	fg_mol_features = []
	num_s = len(s_indices)
	num_q = len(q_indices)
	node_list = []
	for mol_id in range(num_s):
		current_graph = list_graph_support[mol_id]
		num_nodes = current_graph.num_nodes()
		node_list.append(num_nodes)
	for mol_id in range(num_q):
		current_graph = list_graph_query[mol_id]
		num_nodes = current_graph.num_nodes()
		node_list.append(num_nodes)

	list_graph_features = []
	# support samples
	list_graph_features, fg_mol_features, global_id, fg_mol_features, src = handling_support_samples(
		list_graph_features, s_indices,
		num_s, list_graph_support, list_smiles, fg_mol_features, src, global_id)
	# query samples
	num_q = len(q_indices)
	list_graph_features, fg_mol_features, global_id, fg_mol_features, src = handling_query_samples(list_graph_features,
																								   q_indices,
																								   num_q,
																								   list_graph_query,
																								   list_smiles,
																								   fg_mol_features, src,
																								   global_id)
 
	edge_features = []
	mol_start_id = sum(node_list)
	for index_ in range(len(node_list)):  # each graph
		current_num_node = node_list[index_]
		for _ in range(current_num_node):
			dst.append(mol_start_id)
			edge_features.append(0)
		mol_start_id += 1
		mol_feature = list_graph_features[index_]
		fg_mol_features.append(mol_feature)
	end_mol_id = mol_start_id

	return src, dst, end_mol_id, node_list, fg_mol_features, edge_features

def build_context_graph(X, list_p, num_nodes_context, list_smiles, graph_feature_tensor, graph_labels,
						list_graph_support, list_label_support,
						list_graph_query, list_label_query, s_indices, q_indices):
	src = []; dst = []

	list_p = np.array(list_p)
	src, dst, end_mol_id, node_list, fg_mol_features, edge_type = build_FG_Mols(X, list_p, num_nodes_context,
																				list_smiles, graph_feature_tensor,
																				graph_labels, list_graph_support,
																				list_label_support,
																				list_graph_query, list_label_query,
																				s_indices, q_indices)
	fg_mol_features = torch.stack(fg_mol_features)
	fg_mol_pro_features = torch.cat((fg_mol_features, X), dim=0)  # shape: (4, 2)

	num_nodes_context = end_mol_id + args.num_tasks

	g_context = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=num_nodes_context)
	g_context.ndata['x'] = fg_mol_pro_features

	return g_context

from molecules import MoleculeDataset
def LoadData(samples_all, DATASET_NAME):
	return MoleculeDataset(samples_all, DATASET_NAME)

class GraphClassificationDataset:
	def __init__(self):
		self.graph_lists = []  # A list of DGLGraph objects
		self.graph_labels = []
		self.subgraphs = []
	def add(self, g):
		self.graph_lists.append(g)
	def __len__(self):
		return len(self.graphs)
	def __getitem__(self, i):
		return self.graphs[i], self.labels[i], self.subgraphs[i]

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Experiments")

	parser.add_argument("--device", default="cuda:0", help="GPU ids")
	parser.add_argument("--dataset", default="SIDER", help="Dataset")
	parser.add_argument("--model", default="Mainmodel", help="GNN Model")
	parser.add_argument("--k_shot", type=int, default=5)
	parser.add_argument("--norm", type=int, default=1 )
	parser.add_argument("--num_tasks", type=int, default=12)
	parser.add_argument("--num_test_p", type=int, default=3)
	parser.add_argument("--sub_dataset", default="_", help="GNN Model")

	parser.add_argument("--num_eposides", type=int, default=1000)
	parser.add_argument("--num_query", type=int, default=10)
	parser.add_argument("--num_features", type=int, default=64)
	parser.add_argument("--num_classes", type=int, default=1)
	parser.add_argument("--hidden_dim", type=int, default=64)
	parser.add_argument("--run_times", type=int, default=1)
	parser.add_argument("--batch_size", type=int, default=1)   
	parser.add_argument("--lr_inner", type=float, default=1e-3, help="inner learning rate")
	parser.add_argument("--lr_outer", type=float, default=1e-3, help="outer learning rate")
	parser.add_argument("--tasks_per_epoch", type=int, default= 16 )
	parser.add_argument("--fn_epoches", type=int, default=1000)


	parser.add_argument("--k_transition", type=int, default =1 ) 
	parser.add_argument("--d_transfer", type=int, default= 32)
	parser.add_argument("--num_features_org", type=int, default = 64) #
	
	
	args = parser.parse_args()
	print(args)
	device = torch.device(args.device)
	main()