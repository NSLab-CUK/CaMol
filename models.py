import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch_geometric.nn import global_mean_pool
# from torch_scatter import scatter_mean, scatter_add, scatter_std
from dgl.nn.pytorch.glob import SumPooling
from dgl.nn import Set2Set
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F

num_atom_type = 120  
num_chirality_tag = 3

num_bond_type = 6  
num_bond_direction = 3


class MLP(torch.nn.Module):
	def __init__(self, num_features, num_classes, dims=16):
		super(MLP, self).__init__()
		self.mlp = torch.nn.Sequential(
			torch.nn.Linear(num_features, dims),
			torch.nn.ReLU(),
			torch.nn.Linear(dims, num_classes))

	def forward(self, x):
		x = self.mlp(x)
		return x



class GIN(nn.Module):
	def __init__(self, input_dim, hidden_dim=64):
		super().__init__()
		self.ginlayers = nn.ModuleList()
		# self.batch_norms = nn.ModuleList()
		num_layers = 3
		for layer in range(num_layers - 1):  # excluding the input layer
			if layer == 0:
				mlp = MLP(input_dim, hidden_dim, hidden_dim)
			else:
				mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
			self.ginlayers.append(
				GINConv(mlp, aggregator_type='sum', learn_eps=False))  # set to True if learning epsilon

	def forward(self, g, h):
		for i, layer in enumerate(self.ginlayers):
			h = layer(g, h)
			h = F.leaky_relu(h)
		return h


class GINELayer(nn.Module):
	def __init__(self, in_feats, out_feats):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(in_feats, out_feats),
			nn.LeakyReLU(0.01),  # nn.ReLU(),
			nn.Linear(out_feats, out_feats)
		)
 

	def forward(self, g, node_feat, edge_feat):
		with g.local_scope():
			g.ndata['h'] = node_feat
			g.edata['e'] = edge_feat   

			def message_func(edges):
				return {'m': edges.src['h'] + edges.data['e']}  

			g.update_all(message_func, fn.sum('m', 'neigh'))
			h = (1 + 1.0) * g.ndata['h'] + g.ndata['neigh']
			return self.mlp(h)


class GINEModel(nn.Module):
	def __init__(self, in_feats, hidden_dim, num_classes=1):
		super().__init__()
		self.layer1 = GINELayer(in_feats, hidden_dim)
		self.layer2 = GINELayer(hidden_dim, hidden_dim)
		self.layer3 = GINELayer(hidden_dim, hidden_dim)
	def forward(self, g, nfeat, efeat):
		h = self.layer1(g, nfeat, efeat)
		h = self.layer2(g, h, efeat)
		h = self.layer3(g, h, efeat)
		g.ndata['h'] = h
		return h
 


class Meta_model(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.in_dim = args.hidden_dim
		self.hidden_dim = args.hidden_dim
		# self.num_layers = args.num_layers

		self.k_shot = args.k_shot  # 5
		self.num_query = args.num_query  # 10
		self.num_tasks = args.num_tasks  # 12
		self.num_samples = self.k_shot * 2 + self.num_query  # 20
		self.num_classes = args.num_classes
		self.concat_dim = self.hidden_dim * 4

		self.MLP = nn.Sequential(
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.LeakyReLU(0.01),  # nn.ReLU(),
			nn.Linear(self.hidden_dim, self.hidden_dim))
 

		self.compressor = nn.Sequential(
			nn.Linear(self.concat_dim, self.hidden_dim),
			nn.LeakyReLU(0.01),  # nn.ReLU(),
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.LeakyReLU(0.01),  # nn.ReLU(),
			nn.Linear(self.hidden_dim, 1))

		self.Encoder_GIN = GIN(self.in_dim, self.hidden_dim)
		self.Encoder_GINE = GINEModel(self.in_dim, self.hidden_dim)
 
		self.transfer_d = nn.Linear(args.num_features_org, args.d_transfer, bias=False)
		print(f"Loading pre-trained model .pt  ... ")
		self.model = torch.load('pre_trained/pre_training_v1_GIN_64_5_1.pt', map_location=args.device)
		for p in self.model.parameters():
			p.requires_grad = True
		self.MLP2_1d = nn.Sequential(
			nn.Linear(2 * self.hidden_dim, self.hidden_dim),
			nn.LeakyReLU(0.01),  #nn.ReLU(),
			nn.Linear(self.hidden_dim, self.hidden_dim))
		self.e_embed = nn.Embedding(10, self.in_dim)

		self.predict = nn.Sequential(
			nn.Linear(self.concat_dim, self.hidden_dim),
			nn.LeakyReLU(0.01),
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.LeakyReLU(0.01),
			nn.Linear(self.hidden_dim, 1 ))
 
		self.sig = nn.Sigmoid()

	def forward(self, task_id, batch_graph_support, batch_support_x, batch_graph_query, batch_query_x, batch_context,
				batch_context_x, edge_context_feature,
				 device, batch_size,flatten_batch_subgraphs_s, x_subs_s, flatten_batch_subgraphs_q, x_subs_q, mode='inner'):

		self.task_id = task_id;
		self.batch_size = batch_size;
		self.device = device
		nodes_list_support = batch_graph_support.batch_num_nodes();
		nodes_list_query = batch_graph_query.batch_num_nodes()
 

		#Context graph encoder
		edge_context_feature = self.e_embed(edge_context_feature.long()).squeeze(1)
		h_context = self.Encoder_GINE(batch_context, batch_context_x, edge_context_feature)

		# s-cgib:
		batch_support_x = self.transfer_d(batch_support_x); x_subs_s = self.transfer_d(x_subs_s)
		batch_query_x = self.transfer_d(batch_query_x); x_subs_q = self.transfer_d(x_subs_q)

		h_support, _, _, _ = self.model.extract_features(nodes_list_support, batch_graph_support, batch_support_x, flatten_batch_subgraphs_s, x_subs_s, device)  # n 2d
		h_query, _, _, _ = self.model.extract_features(nodes_list_query, batch_graph_query, batch_query_x, flatten_batch_subgraphs_q, x_subs_q, device)  # n 2d
		
 
		###

		sup_pre = None; causal_pre = None; int_pre = None; random_pre = None

 
		h_support_out = self.add_context_mol_support(task_id, nodes_list_support, nodes_list_query, h_support,h_query, h_context, device)
		causal_support_samples, noisy_support_samples, support_mask_prob = self.atom_masking(h_support_out,nodes_list_support,device)


		sup_pre, causal_pre, int_pre, random_pre = self.prediction(h_support_out, batch_graph_support,
																	causal_support_samples, noisy_support_samples)

		return sup_pre, causal_pre, int_pre, random_pre
 

	def prediction(self, h_support_out, batch_graph_support, 
				   causal_support_samples, noisy_support_samples):
 
		batch_graph_support.ndata['x'] = h_support_out;
		support_samples_readout = dgl.sum_nodes(batch_graph_support, 'x')
		sup_pre = self.predict(support_samples_readout)

 
		batch_graph_support.ndata['x'] = causal_support_samples;
		causal_support_samples_readout = dgl.sum_nodes(batch_graph_support, 'x')
		causal_pre = self.predict(causal_support_samples_readout)
 
		batch_graph_support.ndata['x'] = noisy_support_samples;
		h_neg_support_readout = dgl.sum_nodes(batch_graph_support, 'x')
		random_pre = self.predict(h_neg_support_readout)
 
		noise = torch.flip(support_samples_readout, dims=[0])

		h_int = causal_support_samples_readout + noise  

		int_pre = self.predict(h_int)

		return sup_pre, causal_pre, int_pre, random_pre

	def atom_masking(self, graph_features, nodes_list, device):
		causal_features = torch.tensor(()).to(device)
		noisy_features = torch.tensor(()).to(device)
		p_all = torch.tensor(()).to(device)
		z = len(nodes_list)
		graph_feature_split = torch.split(graph_features, tuple(nodes_list))
		for i in range(z):
			features = graph_feature_split[i]

			lambda_pos, p = self.compress(features, device)
			lambda_pos = lambda_pos.reshape(-1, 1)
			lambda_neg = 1 - lambda_pos

			static_node_feature = features.clone().detach()
			node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim=0)

			noisy_node_feature_mean = lambda_pos * features + lambda_neg * node_feature_mean
			noisy_node_feature_std = lambda_neg * node_feature_std
			noisy_node_feature = noisy_node_feature_mean + torch.rand_like(
				noisy_node_feature_mean) * noisy_node_feature_std
			causal_features = torch.cat((causal_features, noisy_node_feature), 0)

			noisy_info = lambda_neg * features
			noisy_features = torch.cat((noisy_features, noisy_info), 0)

			p_all = torch.cat((p_all, p), 0)
		return causal_features, noisy_features, p_all

	def compress(self, graph_features, device):
		p = self.compressor(graph_features)
		temperature = 1.0
		bias = 0.0 + 0.0001
		eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
		gate_inputs = torch.log(eps) - torch.log(1 - eps)
		gate_inputs = gate_inputs.to(device)
		gate_inputs = (gate_inputs + p) / temperature
		gate_inputs = torch.sigmoid(gate_inputs).squeeze()
		return gate_inputs, p

	def add_context_mol_support(self, task_id, nodes_list_support, nodes_list_query, h_support, h_query, h_context,
								device):

		support_query_node_list = torch.cat((nodes_list_support, nodes_list_query), dim=0)
		mol_pro_context_node_list = torch.tensor([self.num_samples, self.num_tasks]).to(device)  # [20, 12]

		tot_node_list = torch.cat((support_query_node_list, mol_pro_context_node_list), dim=0)

		h_context_tuple = torch.split(h_context, tuple(tot_node_list))
		len_node_list_all = len(tot_node_list)

		h_context_property = h_context_tuple[len_node_list_all - 1];
		h_context_molecules = h_context_tuple[len_node_list_all - 2]
		set_h_support = torch.split(h_support, tuple(nodes_list_support))

		# support samples
		h_support_out = torch.tensor(()).to(device)
		for k in range(len(nodes_list_support)):
			mol_support = set_h_support[k]
			fg_context = h_context_tuple[k]
			h_mol_fg = torch.cat((mol_support, fg_context), dim=1)
 

			h_mol_context = h_context_molecules[k];
			h_mol_context_repeated = h_mol_context.unsqueeze(0).repeat(h_mol_fg.shape[0], 1)

			pro_task = h_context_property[task_id];
			 
			pro_task_repeated = pro_task.unsqueeze(0).repeat(h_mol_fg.shape[0], 1)

			result = torch.cat((h_mol_fg, h_mol_context_repeated, pro_task_repeated), dim=1)

			h_support_out = torch.cat((h_support_out, result), 0)
		return h_support_out

	def add_context_mol_query(self, task_id, nodes_list_support, nodes_list_query, h_support, h_query, h_context, device):

		support_query_node_list = torch.cat((nodes_list_support, nodes_list_query), dim=0)

		mol_pro_context_node_list = torch.tensor([self.num_samples, self.num_tasks]).to(device)  # [20, 12]

		tot_node_list = torch.cat((support_query_node_list, mol_pro_context_node_list), dim=0)

		h_context_tuple = torch.split(h_context, tuple(tot_node_list))
		len_node_list_all = len(tot_node_list)

		h_context_property = h_context_tuple[len_node_list_all - 1];
		h_context_molecules = h_context_tuple[len_node_list_all - 2]
		set_h_query = torch.split(h_query, tuple(nodes_list_query))

		current_ind = 0
		for k in range(len(nodes_list_support)):
			current_ind += 1
		# query samples
		h_query_out = torch.tensor(()).to(device)
		for k in range(len(nodes_list_query)):
			mol_query = set_h_query[k]
			fg_context = h_context_tuple[current_ind]
			h_mol_fg = torch.cat((mol_query, fg_context), dim=1)
 

			h_mol_context = h_context_molecules[current_ind] 
			h_mol_context_repeated = h_mol_context.unsqueeze(0).repeat(h_mol_fg.shape[0], 1)

			pro_task = h_context_property[task_id] 
		 
			pro_task_repeated = pro_task.unsqueeze(0).repeat(h_mol_fg.shape[0], 1)

			out = torch.cat((h_mol_fg, h_mol_context_repeated, pro_task_repeated), dim=1)

			h_query_out = torch.cat((h_query_out, out), 0)

			current_ind += 1
		return h_query_out

	def loss(self, scores, targets):
		loss = nn.BCEWithLogitsLoss()  
		l = loss(scores.float(), targets.float())
		return l

class Mainmodel(nn.Module):
	def __init__(self, args, in_dim, hidden_dim, num_layers, num_heads, k_transition, encoder):
		super().__init__()
		self.tau = 1.0
		self.recons_type = args.recons_type
		self.useAtt = args.useAtt

		self.readout = args.readout_f
		self.hidden_dim = hidden_dim
		self.k_transition = k_transition
		self.fc1 = torch.nn.Linear(hidden_dim, 1)

		self.in_dim = args.d_transfer
		self.transfer_d = nn.Linear(in_dim, self.in_dim, bias=False)

		self.embedding_h = nn.Linear(self.in_dim, hidden_dim, bias=False)
		self.attn_layer = nn.Linear(self.hidden_dim * 2, 1)
		self.reduce_d = torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim)
		self.device = args.device
		self.s2s = Set2Set(hidden_dim, 2, 1)

		self.reconstructX = nn.Sequential(
			nn.Linear(self.hidden_dim, self.in_dim))
		self.MLP = nn.Sequential(
			nn.Linear(2 * self.hidden_dim, self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, self.hidden_dim))
		if encoder == "GIN":
			self.Encoder1 = GIN(self.in_dim, hidden_dim)
			self.Encoder2 = GIN(self.in_dim, hidden_dim)
 
		else:
			print("Bug there is no pre-defined Encoders")
			raise SystemExit()

		self.compressor = nn.Sequential(
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.BatchNorm1d(self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, 1))

	def compress(self, graph_features, device):
		p = self.compressor(graph_features)
		temperature = 1.0
		bias = 0.0 + 0.0001  
		eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
		gate_inputs = torch.log(eps) - torch.log(1 - eps)
		gate_inputs = gate_inputs.to(device)
		gate_inputs = (gate_inputs + p) / temperature
		gate_inputs = torch.sigmoid(gate_inputs).squeeze()
		return gate_inputs, p

	def sim(self, z1: torch.Tensor, z2: torch.Tensor):
		z1 = F.normalize(z1)
		z2 = F.normalize(z2)
		return torch.mm(z1, z2.t())

	def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
		device = z1.device
		self.tau = 1
		num_nodes = z1.size(0)  # 32
		num_batches = (num_nodes - 1) // batch_size + 1
		f = lambda x: torch.exp(x / self.tau)
		indices = torch.arange(0, num_nodes).to(device)
		losses = []

		for i in range(num_batches):
			mask = indices[i * batch_size:(i + 1) * batch_size]
			refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
			between_sim = f(self.sim(z1[mask], z2))  # [B, N]

			losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
									 / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size:(
																													  i + 1) * batch_size].diag())))
		ret = torch.cat(losses)
		return ret.mean()

	def compression(self, nodes_list, device):
		epsilon = 0.0000001
		noisy_node_feature_all = torch.tensor(()).to(device)
		p_all = torch.tensor(()).to(device)
		KL_tensor_all = torch.tensor(()).to(device)

		z = len(nodes_list)
		graph_feature_split = torch.split(self.graph_features, tuple(nodes_list))
		for i in range(z):
			features = graph_feature_split[i]

			lambda_pos, p = self.compress(features, device)
			lambda_pos = lambda_pos.reshape(-1, 1)
			lambda_neg = 1 - lambda_pos

			static_node_feature = features.clone().detach()
			node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim=0)
			noisy_node_feature_mean = lambda_pos * features + lambda_neg * node_feature_mean
			noisy_node_feature_std = lambda_neg * node_feature_std
			noisy_node_feature = noisy_node_feature_mean + torch.rand_like(
				noisy_node_feature_mean) * noisy_node_feature_std

			noisy_node_feature_all = torch.cat((noisy_node_feature_all, noisy_node_feature), 0)

			p_all = torch.cat((p_all, p), 0)

			KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2) + torch.sum(
				((noisy_node_feature_mean - node_feature_mean) / (node_feature_std + epsilon)) ** 2, dim=0)
			KL_tensor_all = torch.cat((KL_tensor, KL_tensor), 0)
		return noisy_node_feature_all, p_all, KL_tensor_all

	def forward(self, batch_g, batch_x, flatten_batch_subgraphs, batch_logMs, x_subs, current_epoch, edge_index,
				k_transition, device, batch_size=16):
		self.batch_size = batch_size
		nodes_list = batch_g.batch_num_nodes()
		self.device = device

		batch_x = self.transfer_d(batch_x)
		x_subs = self.transfer_d(x_subs)

		interaction_map, KL_tensor, noisy_node_feature, graph_features_readout = self.extract_features(nodes_list,
																									   batch_g, batch_x,
																									   flatten_batch_subgraphs,
																									   x_subs, device)

		interaction_map = self.MLP(interaction_map)

		# 6. KL upper bound
		KL_Loss = torch.mean(KL_tensor)

		# 7. Contrastive loss
		if self.readout == "sum":
			batch_g.ndata['h'] = noisy_node_feature
			noisy_node_feature_2 = dgl.sum_nodes(batch_g, 'h')  # [nb d]
		else:
			noisy_node_feature_2 = self.s2s(batch_g, noisy_node_feature)  # [nb , 2d]
		contrastive_loss = self.batched_semi_loss(noisy_node_feature_2, graph_features_readout, self.batch_size)

		# 8. Reconstruction loss
		if self.recons_type == 'adj':
			reconstruction_loss = self.loss_recon_adj(interaction_map, batch_g)
		elif self.recons_type == 'logM':
			reconstruction_loss = self.loss_recon(interaction_map, batch_logMs, nodes_list)
		else:
			reconstruction_loss = -1.0
 
		return None, KL_Loss, contrastive_loss, reconstruction_loss

	def extract_features(self, nodes_list, batch_g, batch_x, flatten_batch_subgraphs, x_subs, device):
 
		graph_features = self.Encoder1(batch_g, batch_x)
 
		subgraphs_features = self.Encoder2(flatten_batch_subgraphs, x_subs)
 
		self.graph_features = graph_features   
		self.subgraphs_features = subgraphs_features   

		# Pooling nb x 2d
		if self.readout == "sum":
			batch_g.ndata['h'] = self.graph_features
			graph_features_readout = dgl.sum_nodes(batch_g, 'h')  # [nb d]
		else:
			graph_features_readout = self.s2s(batch_g, self.graph_features)  # [nb 2d]

		# 4. Compression p: preserve_rate
		noisy_node_feature, p, KL_tensor = self.compression(nodes_list, device)

		# 5. Core - subgraphs
		flatten_batch_subgraphs.ndata['h'] = self.subgraphs_features
		subgraphs_features_readout = dgl.sum_nodes(flatten_batch_subgraphs, 'h')
		interaction_map = torch.cat((noisy_node_feature, subgraphs_features_readout), -1)

		# Attention-based interaction
		if self.useAtt:
			subgs_att = torch.tensor(()).to(device)
			if self.readout == "sum":
				batch_g.ndata['h'] = noisy_node_feature
				noisy_node_feature_readout = dgl.sum_nodes(batch_g, 'h')  # [nb d]
			else:
				noisy_node_feature_readout = self.s2s(batch_g, noisy_node_feature)  # nb x 2d
				noisy_node_feature_readout = self.reduce_d(noisy_node_feature_readout)

			subgraphs_features_readout_split = torch.split(subgraphs_features_readout, tuple(nodes_list))
			z = len(subgraphs_features_readout_split)
			for i in range(z):
				noisy_node_feature_readout_cp = noisy_node_feature_readout[i].repeat(nodes_list[i], 1)
				interaction = torch.cat((noisy_node_feature_readout_cp, subgraphs_features_readout_split[i]),
										-1)  # x[n, d]     [n, d] -->  [n  2d]

				layer_atten = self.attn_layer(interaction)
				layer_atten = F.softmax(layer_atten, dim=0)
				a = subgraphs_features_readout_split[i] * layer_atten  # [n , d]
				subgs_att = torch.cat((subgs_att, a), 0)
			interaction_map = torch.cat((noisy_node_feature, subgs_att), -1)
		return interaction_map, KL_tensor, noisy_node_feature, graph_features_readout

	#######################################################		 
	def loss(self, scores, targets):
		loss = nn.BCELoss()
		l = loss(scores.float(), targets.float())
		return l
	def loss_X(self, batch_g, interaction_map):
		interaction_map_X = self.reconstructX(interaction_map)
		loss = F.mse_loss(interaction_map_X, batch_g.ndata['x'])
		return loss

	def loss_recon_adj(self, interaction_map, org_graph, batch_size=16):
		row_num, col_num = interaction_map.size()
		adj = org_graph.adj().to_dense()
		recon_interaction_map = torch.mm(interaction_map, interaction_map.t())

		loss = torch.sum((recon_interaction_map - adj) ** 2) / (row_num)
		return loss

	def loss_recon(self, interaction_map, trans_logM, nodes_list):

		sp_interaction_map = torch.split(interaction_map, tuple(nodes_list))
		loss = 0
		z = len(nodes_list)
		for k in range(z):
			h = torch.mm(sp_interaction_map[k], sp_interaction_map[k].t()).to(self.device)
			row_num, col_num = h.size()
			for i in range(self.k_transition):
				loss += torch.sum(((h - (torch.FloatTensor(trans_logM[k][i])).to(self.device)) ** 2)) / (
							row_num * col_num)
		loss = loss / (self.k_transition)
		return loss

class Mainmodel_continue(nn.Module):
	def __init__(self, args, in_dim, hidden_dim, num_layers, num_heads, k_transition, num_classes, cp_filename,
				 encoder):
		super().__init__()
		self.tau = 1.0
		self.readout = args.readout_f
		self.s2s = Set2Set(hidden_dim, 2, 1)
		self.s2s_rev = Set2Set(in_dim, 2, 1)
		# if args.transfer_mode ==1:
		self.in_dim = args.d_transfer
		# else:
		# 	self.in_dim  = in_dim
		self.transfer_d = nn.Linear(in_dim, self.in_dim, bias=False)
		self.recons_type = args.recons_type

		self.batch_size = args.batch_size
		self.useAtt = args.useAtt
		self.embedding_h = nn.Linear(self.in_dim, hidden_dim, bias=False)

		self.hidden_dim = hidden_dim
		self.k_transition = k_transition
		self.reduce_d = torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim)
		self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)
		self.num_nodes = -1
		self.device = args.device
		self.r_transfer_d = nn.Sequential(
			nn.Linear(2 * self.hidden_dim, self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, in_dim * 2))

		if args.task == "graph_regression":
			self.predict = nn.Sequential(
				nn.Linear(2 * self.hidden_dim, self.hidden_dim),
				nn.ReLU(),
				nn.Linear(self.hidden_dim, 1)
			)
		elif args.task == "graph_classification":
			self.predict = nn.Sequential(
				nn.Linear(2 * self.hidden_dim, self.hidden_dim),
				nn.ReLU(),
				nn.Linear(self.hidden_dim, num_classes)
			)
		else:
			print(f"checking mainmodel_finetuning task ...")

		self.MLP = nn.Sequential(
			nn.Linear(2 * self.hidden_dim, self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, self.hidden_dim))

		if encoder == "GIN":
			self.Encoder1 = GIN(self.in_dim, hidden_dim)
			self.Encoder2 = GIN(self.in_dim, hidden_dim)
 
		else:
			print("Bug there is no pre-defined Encoders")
			raise SystemExit()

		# print(f"Loading pre-trained model .pt (Mainmodel_continue) ... ")
		self.model = torch.load(cp_filename, map_location=args.device)
		for p in self.model.parameters():
			p.requires_grad = True

		self.compressor = nn.Sequential(
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.BatchNorm1d(self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, 1))
		self.reconstructX = nn.Sequential(
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, in_dim))

	def compress(self, graph_features, device):
		p = self.compressor(graph_features)
		temperature = 1.0
		bias = 0.0 + 0.0001  # If bias is 0, we run into problems
		eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
		gate_inputs = torch.log(eps) - torch.log(1 - eps)
		gate_inputs = gate_inputs.to(device)
		gate_inputs = (gate_inputs + p) / temperature
		gate_inputs = torch.sigmoid(gate_inputs).squeeze()
		return gate_inputs, p

	def sim(self, z1: torch.Tensor, z2: torch.Tensor):
		z1 = F.normalize(z1)
		z2 = F.normalize(z2)
		return torch.mm(z1, z2.t())

	def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
		device = z1.device
		self.tau = 1
		num_nodes = z1.size(0)  # 32
		num_batches = (num_nodes - 1) // batch_size + 1
		f = lambda x: torch.exp(x / self.tau)
		indices = torch.arange(0, num_nodes).to(device)
		losses = []

		for i in range(num_batches):
			mask = indices[i * batch_size:(i + 1) * batch_size]
			refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
			between_sim = f(self.sim(z1[mask], z2))  # [B, N]

			losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
									 / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size:(
																													  i + 1) * batch_size].diag())))
		ret = torch.cat(losses)
		return ret.mean()

	def compression(self, nodes_list, device):
		epsilon = 0.0000001
		noisy_node_feature_all = torch.tensor(()).to(device)
		p_all = torch.tensor(()).to(device)
		KL_tensor_all = torch.tensor(()).to(device)

		z = len(nodes_list)
		graph_feature_split = torch.split(self.graph_features, tuple(nodes_list))
		for i in range(z):
			features = graph_feature_split[i]

			lambda_pos, p = self.compress(features, device)
			lambda_pos = lambda_pos.reshape(-1, 1)
			lambda_neg = 1 - lambda_pos

			static_node_feature = features.clone().detach()
			node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim=0)
			noisy_node_feature_mean = lambda_pos * features + lambda_neg * node_feature_mean
			noisy_node_feature_std = lambda_neg * node_feature_std
			noisy_node_feature = noisy_node_feature_mean + torch.rand_like(
				noisy_node_feature_mean) * noisy_node_feature_std

			noisy_node_feature_all = torch.cat((noisy_node_feature_all, noisy_node_feature), 0)

			p_all = torch.cat((p_all, p), 0)

			KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2) + torch.sum(
				((noisy_node_feature_mean - node_feature_mean) / (node_feature_std + epsilon)) ** 2, dim=0)
			KL_tensor_all = torch.cat((KL_tensor, KL_tensor), 0)
		return noisy_node_feature_all, p_all, KL_tensor_all

	def forward(self, batch_g, batch_x, flatten_batch_subgraphs, batch_logMs, x_subs, current_epoch, edge_index,
				k_transition, device, batch_size=16):
		self.batch_size = batch_size
		nodes_list = batch_g.batch_num_nodes()
		self.device = device

		batch_x = self.transfer_d(batch_x)
		x_subs = self.transfer_d(x_subs)

		interaction_map, KL_tensor, noisy_node_feature, graph_features_readout = self.model.extract_features(nodes_list,
																											 batch_g,
																											 batch_x,
																											 flatten_batch_subgraphs,
																											 x_subs,
																											 device)

		interaction_map = self.MLP(interaction_map)

		# 6. KL upper bound
		KL_Loss = torch.mean(KL_tensor)

 
		if self.readout == "sum":
			batch_g.ndata['h'] = noisy_node_feature
			noisy_node_feature_2 = dgl.sum_nodes(batch_g, 'h')  # [nb d]
		else:
			noisy_node_feature_2 = self.s2s(batch_g, noisy_node_feature)  # [nb , 2d]
		contrastive_loss = self.batched_semi_loss(noisy_node_feature_2, graph_features_readout, self.batch_size)
 
		if self.recons_type == 'adj':
			reconstruction_loss = self.loss_recon_adj(interaction_map, batch_g)
		elif self.recons_type == 'logM':
			reconstruction_loss = self.loss_recon(interaction_map, batch_logMs, nodes_list)
		else:
			reconstruction_loss = -1.0

		return None, KL_Loss, contrastive_loss, reconstruction_loss

	def loss_X(self, batch_x_org, interaction_map):

		row_num, col_num = interaction_map.size()
		loss = torch.sum((interaction_map - batch_x_org) ** 2) / (row_num)
		return loss

	def extract_features(self, nodes_list, batch_g, batch_x, flatten_batch_subgraphs, x_subs, device):
 
		graph_features = self.Encoder1(batch_g, batch_x)

		subgraphs_features = self.Encoder2(flatten_batch_subgraphs, x_subs)

		self.graph_features = graph_features  
		self.subgraphs_features = subgraphs_features  

		# Pooling nb x 2d
		if self.readout == "sum":
			batch_g.ndata['h'] = self.graph_features
			graph_features_readout = dgl.sum_nodes(batch_g, 'h')  # [nb d]
		else:
			graph_features_readout = self.s2s(batch_g, self.graph_features)  # [nb 2d]

		noisy_node_feature, p, KL_tensor = self.compression(nodes_list, device)

		flatten_batch_subgraphs.ndata['h'] = self.subgraphs_features
		subgraphs_features_readout = dgl.sum_nodes(flatten_batch_subgraphs, 'h')
		interaction_map = torch.cat((noisy_node_feature, subgraphs_features_readout), -1)

		if self.useAtt:
			subgs_att = torch.tensor(()).to(device)
			if self.readout == "sum":
				batch_g.ndata['h'] = noisy_node_feature
				noisy_node_feature_readout = dgl.sum_nodes(batch_g, 'h')  # [nb d]
			else:
				noisy_node_feature_readout = self.s2s(batch_g, noisy_node_feature)  # nb x 2d
				noisy_node_feature_readout = self.reduce_d(noisy_node_feature_readout)

			subgraphs_features_readout_split = torch.split(subgraphs_features_readout, tuple(nodes_list))
			z = len(subgraphs_features_readout_split)
			for i in range(z):
				noisy_node_feature_readout_cp = noisy_node_feature_readout[i].repeat(nodes_list[i], 1)
				interaction = torch.cat((noisy_node_feature_readout_cp, subgraphs_features_readout_split[i]),
										-1)  # x[n, d]     [n, d] -->  [n  2d]

				layer_atten = self.attn_layer(interaction)
				layer_atten = F.softmax(layer_atten, dim= 0)
				a = subgraphs_features_readout_split[i] * layer_atten  # [n , d]
				subgs_att = torch.cat((subgs_att, a), 0)
			interaction_map = torch.cat((noisy_node_feature, subgs_att), -1)
		return interaction_map, KL_tensor, noisy_node_feature, graph_features_readout


	def loss_recon_adj(self, interaction_map, org_graph, batch_size=16):
		row_num, col_num = interaction_map.size()
		adj = org_graph.adj().to_dense()
		recon_interaction_map = torch.mm(interaction_map, interaction_map.t())

		loss = torch.sum((recon_interaction_map - adj) ** 2) / (row_num)
		return loss

	def loss_recon(self, interaction_map, trans_logM, nodes_list):

		sp_interaction_map = torch.split(interaction_map, tuple(nodes_list))
		loss = 0
		z = len(nodes_list)
		for k in range(z):
			h = torch.mm(sp_interaction_map[k], sp_interaction_map[k].t()).to(self.device)
			row_num, col_num = h.size()
			for i in range(self.k_transition):
				loss += torch.sum(((h - (torch.FloatTensor(trans_logM[k][i])).to(self.device)) ** 2)) / (
							row_num * col_num)
		loss = loss / (self.k_transition)
		return loss
class MLPA(torch.nn.Module):
	def __init__(self, in_feats, dim_h, dim_z):
		super(MLPA, self).__init__()
		self.gcn_mean = torch.nn.Sequential(
			torch.nn.Linear(in_feats, dim_h),
			torch.nn.ReLU(),
			torch.nn.Linear(dim_h, dim_z)
		)

	def forward(self, hidden):
		Z = self.gcn_mean(hidden)
		adj_logits = Z @ Z.T
		return adj_logits

