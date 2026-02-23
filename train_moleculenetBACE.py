
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
loss_func = nn.CrossEntropyLoss()
from metrics import accuracy_TU, MAE
def cosinSim(x_hat):
    x_norm = torch.norm(x_hat, p=2, dim=1)
    nume = torch.mm(x_hat, x_hat.t())
    deno = torch.ger(x_norm, x_norm)
    cosine_similarity = nume / deno
    return cosine_similarity


from ogb.graphproppred import Evaluator
def process_diff(batch_adj, batch_size):
    list_batch = []
    max_size = 0
    #print(f"batch_adj[i]: {batch_adj[0]}")
    for i in range(batch_size):
        size= batch_adj[i].size(dim=1)
        if size> max_size:
            max_size = size
    
    p2d = (0,2,0,2) # pad last dim by 1 on each side
    for i in range(batch_size):
        diff= max_size - batch_adj[i].size(dim=1)
        if diff != max_size:
            p2d = (0,diff,0,diff) # pad last dim by 1 on each side
            batch_adj[i] = F.pad(batch_adj[i], p2d, "constant", 0) 
            #print(f"batch_adj[i]: {batch_adj[i].size()}")
            list_batch.append(batch_adj[i])
    return torch.stack(batch_adj, dim=0)
    #raise SystemExit()
import dgl
from itertools import chain
from sklearn.neighbors import KernelDensity
from scipy.integrate import simps
def compute_js_divergence1d(set_a, set_b, bandwidth=0.5):
    """
    Compute the Jensen-Shannon Divergence between two 1D datasets using KDE.
    """
    # Convert sets to numpy arrays
    A = np.array(set_a.detach()).reshape(-1, 1)
    B = np.array(set_b.detach()).reshape(-1, 1)
    
    # Fit Kernel Density Estimation models to both sets
    kde_A = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(A)
    kde_B = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(B)
    
    # Create a grid of points in the domain of the data (you can adjust the number of points)
    sample_points = np.linspace(min(min(A), min(B)), max(max(A), max(B)), 1000).reshape(-1, 1)
    
    # Evaluate the log density of both sets at these sample points
    log_density_A = kde_A.score_samples(sample_points)
    log_density_B = kde_B.score_samples(sample_points)
    
    # Convert log densities to actual densities
    density_A = np.exp(log_density_A)
    density_B = np.exp(log_density_B)
    
    # Compute the midpoint distribution M = 0.5 * (density_A + density_B)
    density_M = 0.5 * (density_A + density_B)
    
    # Compute KL Divergence from A to M and from B to M
    kl_A_M = np.sum(density_A * (log_density_A - np.log(density_M)))
    kl_B_M = np.sum(density_B * (log_density_B - np.log(density_M)))
    
    # Compute the Jensen-Shannon Divergence
    js_divergence = 0.5 * (kl_A_M + kl_B_M)
    print("Jensen-Shannon Divergence:", js_divergence)
def compute_js_divergence2d(tensor_a, tensor_b, bandwidth=0.5, grid_size=100):
    # Convert tensors to numpy arrays
    X = tensor_a.detach().numpy()
    Y = tensor_b.detach().numpy()
    
    # Fit KDE models for both sets
    kde_X = KernelDensity(bandwidth=bandwidth).fit(X)
    kde_Y = KernelDensity(bandwidth=bandwidth).fit(Y)

    # Define a grid over which to evaluate the density
    x_min, y_min = np.min(np.vstack([X, Y]), axis=0) - 1
    x_max, y_max = np.max(np.vstack([X, Y]), axis=0) + 1
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X_grid, Y_grid = np.meshgrid(x, y)
    grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

    # Evaluate densities on the grid
    log_P = kde_X.score_samples(grid_points).reshape(grid_size, grid_size)
    log_Q = kde_Y.score_samples(grid_points).reshape(grid_size, grid_size)

    # Convert log densities to actual densities
    P = np.exp(log_P)
    Q = np.exp(log_Q)

    # Compute the midpoint distribution M
    M = 0.5 * (P + Q)

    # Calculate KL divergence components for JS divergence
    KL_P_M = simps(simps(P * (log_P - np.log(M)), y), x)
    KL_Q_M = simps(simps(Q * (log_Q - np.log(M)), y), x)

    # Calculate Jensen-Shannon Divergence
    JS_divergence = 0.5 * (KL_P_M + KL_Q_M)
    print("Jensen-Shannon Divergence:", JS_divergence)
# Function to compute KL Divergence from P to Q using KDE
def compute_kl_divergence1ds(p, q, bandwidth=0.5):
    """
    Compute Kullback-Leibler (KL) Divergence from distribution p to distribution q.
    p and q are 1D arrays.
    """
    p = np.array(p).reshape(-1, 1)  # Reshape to 2D for KDE
    q = np.array(q).reshape(-1, 1)
    
    # Fit Kernel Density Estimation (KDE) models to the two distributions
    kde_p = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(p)
    kde_q = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(q)
    
    # Generate a grid of points to evaluate the density on
    sample_points = np.linspace(min(min(p), min(q)), max(max(p), max(q)), 1000).reshape(-1, 1)
    
    # Evaluate the log-density of each set at the sample points
    log_density_p = kde_p.score_samples(sample_points)
    log_density_q = kde_q.score_samples(sample_points)
    
    # Convert log-densities to actual densities
    density_p = np.exp(log_density_p)
    density_q = np.exp(log_density_q)
    
    # Compute KL Divergence from p to q
    kl_div = np.sum(density_p * (log_density_p - np.log(density_q)))
    
    return kl_div
# Function to compute Jensen-Shannon Divergence between two sets of 1D vectors
def compute_kl_divergence(p, q, bandwidth=0.5):
    """
    Compute Kullback-Leibler (KL) Divergence from distribution p to distribution q.
    p and q are multi-dimensional arrays of shape (n_samples, n_features).
    """
    # Fit Kernel Density Estimation (KDE) models to the two distributions
    kde_p = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(p)
    kde_q = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(q)
    
    # Generate a grid of points to evaluate the density (adjust range based on data)
    min_vals = np.min(np.vstack([p, q]), axis=0)
    max_vals = np.max(np.vstack([p, q]), axis=0)
    
    grid_points = np.meshgrid(*[np.linspace(min_val, max_val, 100) for min_val, max_val in zip(min_vals, max_vals)])
    grid_points = np.vstack([g.flatten() for g in grid_points]).T  # Flatten to 2D grid

    # Evaluate the log-density of each distribution on the grid points
    log_density_p = kde_p.score_samples(grid_points)
    log_density_q = kde_q.score_samples(grid_points)
    
    # Convert log-densities to actual densities
    density_p = np.exp(log_density_p)
    density_q = np.exp(log_density_q)
    
    # Compute KL Divergence from p to q
    kl_div = np.sum(density_p * (log_density_p - np.log(density_q)))
    
    return kl_div
def compute_js_divergence(p, q, bandwidth=0.5):
    """
    Compute the Jensen-Shannon Divergence (JS) between two 1D data distributions.
    """
    # Compute KL Divergence from p to q and from q to p
    kl_p_q = compute_kl_divergence(p.detach(), q.detach(), bandwidth)
    kl_q_p = compute_kl_divergence(q.detach(), p.detach(), bandwidth)
    
    # Compute the Jensen-Shannon Divergence (average of the two KL divergences)
    js_divergence = 0.5 * (kl_p_q + kl_q_p)
    print(f"Jensen-Shannon Divergence between set_a and set_b: {js_divergence}")
def compute_js_divergence_high_dim(set_a, set_b, bandwidth=0.5, sample_size=1000):
    # Ensure input is numpy array
    A = np.array(set_a.detach())
    B = np.array(set_b.detach())
    
    # Fit KDE models for both sets
    kde_A = KernelDensity(bandwidth=bandwidth).fit(A)
    kde_B = KernelDensity(bandwidth=bandwidth).fit(B)

    # Sample points from each distribution (use original points if sample_size is larger)
    sample_points = np.vstack([A[np.random.choice(A.shape[0], sample_size, replace=True)],
                               B[np.random.choice(B.shape[0], sample_size, replace=True)]])

    # Evaluate densities for each KDE on these sample points
    log_density_A = kde_A.score_samples(sample_points)
    log_density_B = kde_B.score_samples(sample_points)
    
    # Convert log densities to actual densities
    density_A = np.exp(log_density_A)
    density_B = np.exp(log_density_B)
    
    # Compute midpoint distribution M
    M = 0.5 * (density_A + density_B)
    
    # Calculate KL divergence components for JS divergence
    KL_A_M = np.mean(density_A * (log_density_A - np.log(M)))
    KL_B_M = np.mean(density_B * (log_density_B - np.log(M)))

    # Calculate Jensen-Shannon Divergence
    JS_divergence = 0.5 * (KL_A_M + KL_B_M)
    print(f"Jensen-Shannon Divergence between set_a and set_b: {JS_divergence}")
def train_epoch_graph_classification(args, model, optimizer, device, data_loader, epoch, batch_size):
    model.train()
    
    if args.dataset == "BACE":
        evaluator = Evaluator(name = "ogbg-molbace")
    elif args.dataset == "BBBP":
        evaluator = Evaluator(name = "ogbg-molbbbp")
    elif args.dataset == "Tox21":
        evaluator = Evaluator(name = "ogbg-moltox21")
    elif args.dataset == "ToxCast":
        evaluator = Evaluator(name = "ogbg-moltox21")
    elif args.dataset == "ClinTox":
        evaluator = Evaluator(name = "ogbg-molclintox")
    else:
        evaluator =None 
    epoch_train_auc = 0
    epoch_train_acc = 0
    epoch_train_mae = 0
    targets=torch.tensor([]).to(device)
    scores=torch.tensor([]).to(device)
    epoch_loss = 0
    nb_data = 0
    gpu_mem = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    embeddings = torch.tensor(()).to(device)
    for iter, (batch_graphs, batch_targets, batch_subgraphs, batch_fgs) in enumerate(data_loader):
        count = iter
        batch_targets = batch_targets.to(device)
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['x'].float().to(device)
        edge_index = batch_graphs.edges()

        optimizer.zero_grad()

        flatten_batch_subgraphs = list(chain.from_iterable(batch_subgraphs))
        flatten_batch_subgraphs = dgl.batch(flatten_batch_subgraphs).to(device)
        x_subs = flatten_batch_subgraphs.ndata['x'].float().to(device)

        flatten_batch_fgs = list(chain.from_iterable(batch_fgs))
        flatten_batch_fgs = dgl.batch(flatten_batch_fgs).to(device)
        x_fgs = flatten_batch_fgs.ndata['x'].float().to(device)

        batch_scores, _, _, _ = model.forward(
            batch_graphs, batch_x,
            flatten_batch_subgraphs, x_subs,
            flatten_batch_fgs, x_fgs,
            device, batch_size)
        #batch_targets = batch_targets.unsqueeze(1)
        #print(f"batch_scores {batch_scores.shape} | batch_targets {batch_targets.shape}")
 
        loss = model.loss(batch_scores, batch_targets)
        # batch_scores = batch_scores.long()
        #batch_targets = batch_targets.long()
        #print(f"batch_targets  {batch_targets.dtype} {batch_targets.shape} | batch_scores {batch_scores.dtype} {batch_scores}")  
        #raise SystemExit()
        #loss = criterion(batch_scores, batch_targets)

        targets = torch.cat((targets, batch_targets), 0)
        scores = torch.cat((scores, batch_scores), 0)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()     

    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)

    input_dict = {"y_true": targets, "y_pred": scores}
    epoch_train_auc = evaluator.eval(input_dict)['rocauc']  
    return epoch_loss, epoch_train_auc, optimizer

    for iter, (batch_graphs, batch_targets, batch_subgraphs, _ ) in enumerate(data_loader):
 
        batch_targets = batch_targets.to(device)
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['x'].float().to(device)  # num x feat
 
        edge_index = batch_graphs.edges()
    
        optimizer.zero_grad()
 
        flatten_batch_subgraphs = list(chain.from_iterable(batch_subgraphs))
        # print(flatten_batch_subgraphs)

        flatten_batch_subgraphs  = dgl.batch(flatten_batch_subgraphs).to(device) # batch_subgraphs.to(device)
        x_subs = flatten_batch_subgraphs.ndata['x'].float().to(device)  # num x feat
 
        batch_x  = F.normalize(batch_x)
        x_subs  = F.normalize(x_subs)
        batch_scores, embeds, _, _ = model.forward(batch_graphs,batch_x,flatten_batch_subgraphs , x_subs,  1, edge_index, 2, device, batch_size)
 
        parameters = []
        for parameter in model.parameters():
            parameters.append(parameter.view(-1))


        # batch_targets= batch_targets.squeeze(1)
        # print(f"batch_scores: {batch_scores.shape}|  {batch_scores}")
        # print(f"batch_targets:  {batch_targets.shape}|   {batch_targets}")
        # raise SystemExit()
        loss = model.loss(batch_scores, batch_targets)
        targets = torch.cat((targets, batch_targets), 0)
        scores = torch.cat((scores, batch_scores), 0)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()     
        embeddings = torch.cat((embeddings, batch_scores), 0)
        # print(targets)
        # raise SystemExit()
    X = (targets >=1)
    #print(X)
    #X = X.flatten()
    #print(X)
    
    embeddings_X = embeddings[X]
    embeddings_Y = embeddings[~X]
    #print(embeddings_X.shape); print(embeddings_Y.shape)
    compute_js_divergence(embeddings_X.cpu(),embeddings_Y.cpu())
    #raise SystemExit()
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
   
    input_dict = {"y_true": targets, "y_pred": scores}
    epoch_train_auc = evaluator.eval(input_dict)['rocauc']  
    return epoch_loss, epoch_train_auc, optimizer
from metrics import Fidelity 
def compute_js_divergence(tensor_a, tensor_b, bandwidth=0.5, grid_size=1000):
    # Convert tensors to numpy arrays
    X = tensor_a.detach().numpy()
    Y = tensor_b.detach().numpy()
    
    # Fit KDE models for both sets
    kde_X = KernelDensity(bandwidth=bandwidth).fit(X)
    kde_Y = KernelDensity(bandwidth=bandwidth).fit(Y)

    # Define a grid over which to evaluate the density
    x_min, y_min = np.min(np.vstack([X, Y]), axis=0) - 1
    x_max, y_max = np.max(np.vstack([X, Y]), axis=0) + 1
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X_grid, Y_grid = np.meshgrid(x, y)
    grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

    # Evaluate densities on the grid
    log_P = kde_X.score_samples(grid_points).reshape(grid_size, grid_size)
    log_Q = kde_Y.score_samples(grid_points).reshape(grid_size, grid_size)

    # Convert log densities to actual densities
    P = np.exp(log_P);     Q = np.exp(log_Q) 
    #P =np.power(2, log_P)  ;     Q = np.power(2, log_Q)  #np.power(2, x)

    # Compute the midpoint distribution M
    M = 0.5 * (P + Q)

    # Calculate KL divergence components for JS divergence
    KL_P_M = simps(simps(P * (log_P - np.log(M)), y), x)
    KL_Q_M = simps(simps(Q * (log_Q - np.log(M)), y), x)

    # Calculate Jensen-Shannon Divergence
    JS_divergence = KL_P_M + KL_Q_M
    print("Jensen-Shannon Divergence:", JS_divergence)
# ogbg-molbace
# ogbg-molbbbp
# ogbg-molclintox
# ogbg-molmuv
# ogbg-molpcba
# ogbg-molsider
# ogbg-moltox21
# ogbg-moltoxcast
# ogbg-molhiv
# ogbg-molesol
# ogbg-molfreesolv
# ogbg-mollipo
# ogbg-molchembl
# ogbg-ppa
# ogbg-code2
def evaluate_network(args, model, optimizer, device, data_loader, epoch, batch_size): #(model, device, data_loader, epoch):
    model.eval()
    if args.dataset == "BACE":
        evaluator = Evaluator(name = "ogbg-molbace")
    elif args.dataset == "BBBP":
        evaluator = Evaluator(name = "ogbg-molbbbp")
    else:
        evaluator =None 
    targets=torch.tensor([]).to(device)
    scores=torch.tensor([]).to(device)
    scores_pos =torch.tensor([]).to(device); scores_neg =torch.tensor([]).to(device)
    epoch_test_loss = 0
    epoch_test_auc = 0
    num = 1
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        #for iter, (batch_graphs, batch_targets, _) in enumerate(data_loader):
        for iter, (batch_graphs, batch_targets, batch_subgraphs, batch_fgs) in enumerate(data_loader):
            count = iter
            batch_targets = batch_targets.to(device)
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['x'].float().to(device)
            edge_index = batch_graphs.edges()

            optimizer.zero_grad()

            flatten_batch_subgraphs = list(chain.from_iterable(batch_subgraphs))
            flatten_batch_subgraphs = dgl.batch(flatten_batch_subgraphs).to(device)
            x_subs = flatten_batch_subgraphs.ndata['x'].float().to(device)

            flatten_batch_fgs = list(chain.from_iterable(batch_fgs))
            flatten_batch_fgs = dgl.batch(flatten_batch_fgs).to(device)
            x_fgs = flatten_batch_fgs.ndata['x'].float().to(device)

            batch_scores, batch_scores_pos, batch_scores_neg, _ = model.forward(
                batch_graphs, batch_x,
                flatten_batch_subgraphs, x_subs,
                flatten_batch_fgs, x_fgs,
                device, batch_size)
            #batch_targets = batch_targets.unsqueeze(1)
 
            #batch_targets= batch_targets.squeeze(1)
            loss = model.loss(batch_scores, batch_targets)
            targets = torch.cat((targets, batch_targets), 0)
            scores = torch.cat((scores, batch_scores), 0)
 
            epoch_test_loss += loss.detach().item()

            # scores_pos= torch.cat((scores_pos, batch_scores_pos), 0)
            # scores_neg= torch.cat((scores_neg, batch_scores_neg), 0)
            nb_data += batch_targets.size(0)

            # print(f" batch_scores_pos {batch_scores_pos}")
            # print(f" batch_scores_neg {batch_scores_neg}")
            # raise SystemExit()

        input_dict = {"y_true": targets, "y_pred": scores}
        epoch_test_auc = evaluator.eval(input_dict)['rocauc']  

        epoch_test_loss /= (iter + 1)
 
        # fid_pos, fid_neg , acc_pos, acc_neg = Fidelity(scores, targets, scores_pos, scores_neg)
        # print(f"Fidelity- : {np.round( fid_neg, 4)}  |Fidelity+ :  {np.round( fid_pos, 4)} ") 

    return epoch_test_loss, epoch_test_auc
 