# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:30:25 2022

@author: chunman zuo
"""

import os
import time
import argparse
import torch
import random
import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, pairwise_distances
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, scale
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine

def parameter_setting():
	
	parser      = argparse.ArgumentParser(description='Spatial transcriptomics analysis by HIN')

	parser.add_argument('--inputPath',   '-IP', type = str, default = None,    help='data directory')	
	parser.add_argument('--outPath', '-od', type=str, default = None, help='Output path')

	parser.add_argument('--spotGene',   '-sGene', type = str, default = None,    help='gene neighbors for each spot')	
	parser.add_argument('--spotGroup', '-sGroup', type=str, default = None, help='group neighbors for each spot')
	parser.add_argument('--spotLatent',   '-sLatent', type = str, default = None,    help='Spot latent feaures with 50-dimensional embeddings by AE')
	parser.add_argument('--GeneLatent',   '-GeneLatent', type = str, default = None,    help='Gene latent feaures by a first HIN model')	
	parser.add_argument('--visualFeature', '-vFeature', type=str, default = None, help='Spot visual features with 2048-dimension')
	parser.add_argument('--spatialLocation', '-sLocation', type=str, default = None, help='spot physical location')

	parser.add_argument('--Ligands_exp', '-Ligands_exp', type=str, default = None, help='Expression of ligands per spot')
	parser.add_argument('--Receptors_exp', '-Receptors_exp', type=str, default = None, help='Expression of receptors per spot')

	parser.add_argument('--Cluster_file', '-Cluster_file', type=str, default = None, help='Pre-cluster file for spot')

	parser.add_argument('--cci_pairs', '-cci_pairs', type=int, default = 5304, help='The number of receptors for each spot')
	parser.add_argument('--cci_neighs', '-cci_neighs', type=int, default = 13, help='The number of neighbors for each spot')
	parser.add_argument('--cluster_pre', '-cluster_pre', type=int, default = 7, help='Cell clusters')
	
	parser.add_argument('--visMeasure',   '-visMeas', type = str, default = 'cosine',    help='Calcualte spot visual feature similarity by cosine')	
	parser.add_argument('--rnaMeasure',   '-rnaMeas', type = str, default = 'cosine',    help='Calcualte spot RNA feature similarity by cosine')	
	parser.add_argument('--locMeasure',   '-locMeas', type = str, default = 'euclidean',    help='Calcualte spot location similarity by euclidean')	

	parser.add_argument('--spotGene_exp', '-sGroup_gene', type=str, default = None, help='adjancy matrix for spots based on cell-gene-cell semantics')
	parser.add_argument('--spot_Group',   '-s_Group', type = str, default = None,    help='adjancy matrix for spots based on both cell-group-cell and cell-gene-cell semantics')	
	parser.add_argument('--pos_pair', '-posP', type=str, default = None, help='positive pairs beween spots based on both cell-group-cell and cell-gene-cell semantics')
	parser.add_argument('--PPI_file',   '-PPI_F', type = str, default = None,    help='gene-gene interaction network')

	parser.add_argument('--geneSpot',   '-geneSpot', type = str, default = None,    help='spot neighbors for each gene')	
	parser.add_argument('--geneGroup', '-geneGroup', type=str, default = None, help='group neighbors for each gene')
	parser.add_argument('--Genespotgene', '-Genespotgene', type=str, default = None, help='gene-spot-gene')
	parser.add_argument('--Genelocgene',   '-Genelocgene', type = str, default = None,    help='gene-location-gene')	
	parser.add_argument('--pos_pair_gene', '-pos_pair_gene', type=str, default = None, help='positive pairs between genes')

	parser.add_argument('--weight_decay', type=float, default = 1e-6, help='weight decay')
	parser.add_argument('--eps', type=float, default = 0.01, help='eps')

	parser.add_argument('--Node_type', '-NodeT',type=int, default=3, help='The node type of spatially resolved transcriptomics data, i.e., Spot, gene, location')
	parser.add_argument('--Node_list', '-NodeL',type=list, default=[4727, 3000, 16], help='Three node list of SRT data')
	parser.add_argument('--sample_rate', '-sample_rate',type=list, default=[100,1], help='Three node list of SRT data')

	parser.add_argument('--hidden_dim', '-hd',type=int, default=50, help='same hidden dim features for three node types of data')

	parser.add_argument('--tau', '-tau', type=float, default=0.8)
	parser.add_argument('--feat_drop', '-feat_drop', type=float, default=0.3)
	parser.add_argument('--attn_drop', '-attn_drop', type=float, default=0.5)
	parser.add_argument('--lam', '-lam', type=float, default=0.5)

	parser.add_argument('--max_training', '-maxT', type=int, default=1000, help='Max epoches for training')
	parser.add_argument('--lr', '-lr', type=float, default = 0.002, help='Learning rate')
	parser.add_argument('--lr_cci', '-lr_cci', type=float, default = 0.002, help='Learning rate')
	parser.add_argument('--l2_coef', '-l2_coef', type=float, default=0)
	parser.add_argument('--patience', '-patience', type=int, default=30)

	parser.add_argument('--batch_size_T', '-bT', type=int, default=128, help='Batch size for transcriptomics data')
	parser.add_argument('--epoch_per_test', '-ept', type=int, default=5, help='Epoch per test')
	parser.add_argument('--lr_AET', type=float, default = 8e-05, help='Learning rate for transcriptomics data for AE model')
	parser.add_argument('--lr_AET_F', type=float, default = 8e-06, help='final learning rate for transcriptomics data for AE model')
	parser.add_argument('--max_epoch_T', '-meT', type=int, default=1000, help='Max epoches for transcriptomics data')

	parser.add_argument('--save_emb', '-save_emb',  default=True, action='store_true', help="save ebedding to file")

	parser.add_argument('--graph_model', '-graphModel', type=str, default = 'GAT', help='graph attention model (GAT or GCN)')
	parser.add_argument('--attention_head', '-attentionHead', type=int, default = 2, help='the number of attention heads (GAT or GCN)')
	parser.add_argument('--fusion_type', '-fusionType', type=str, default = "Attention", help='the type of multi-view graph fusion')

	parser.add_argument('--use_cuda', dest='use_cuda', default=True, action='store_true', help=" whether use cuda(default: True)")
	parser.add_argument('--seed', type=int, default=200, help='Random seed for repeat results')

	parser.add_argument('--knn', '-KNN', type=int, default=7, help='K nearst neighbour include itself')

	parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
	parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
	
	return parser

def encode_onehot(labels):
	labels = labels.reshape(-1, 1)
	enc = OneHotEncoder()
	enc.fit(labels)
	labels_onehot = enc.transform(labels).toarray()
	return labels_onehot

def preprocess_features(features):
	"""Row-normalize feature matrix and convert to tuple representation"""
	rowsum = np.array(features.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	features = r_mat_inv.dot(features)
	return features

def normalize_adj(adj):
	"""Symmetrically normalize adjacency matrix."""
	adj = sp.coo_matrix(adj)
	rowsum = np.array(adj.sum(1))
	d_inv_sqrt = np.power(rowsum, -0.5).flatten()
	d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
	d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
	return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
	"""Convert a scipy sparse matrix to a torch sparse tensor."""
	sparse_mx = sparse_mx.tocoo().astype(np.float32)
	indices   = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
	values    = torch.from_numpy(sparse_mx.data)
	shape     = torch.Size(sparse_mx.shape)
	return torch.sparse.FloatTensor(indices, values, shape)

def load_cci_ligand_receptors( args ):

	print("spot location for adjancy")
	spot_loc     = pd.read_table(args.spatialLocation, header = None, index_col = None).values
	dist_loc     = pairwise_distances(spot_loc, metric = args.locMeasure)
	
	sorted_knn    = dist_loc.argsort(axis=1)
	selected_node = []
	#used_spots    = []
	for index in list(range( np.shape(dist_loc)[0] )):
		selected_node.append( sorted_knn[index, :11] )
		#used_spots.extend( sorted_knn[index, :11] )
	selected_node  = torch.LongTensor(selected_node)
	#used_spots     = torch.LongTensor(list(set(used_spots)))

	print("spot-ligand data")
	spots_ligand  = pd.read_table(args.Ligands_exp, header = None, index_col = None).values
	spots_ligand  = torch.FloatTensor(spots_ligand)

	print("spot-receptor data")
	spots_recep   = pd.read_table(args.Receptors_exp, header = None, index_col = None).values
	spots_recep   = torch.FloatTensor(spots_recep)

	#print("cluster data")
	#cluster    = pd.read_table(args.Cluster_file, header = None, index_col = None).values
	#cluster_n  = torch.LongTensor( cluster[:,0] )

	pos   = pd.read_table(args.pos_pair, header = None, index_col = None).values
	pos   = torch.FloatTensor(pos)

	return selected_node, spots_ligand, spots_recep, pos


def load_data_naive( args ):
	# The order of node types: cell, gene, group

	#Spot-RNA
	print("Spot-RNA")
	spot_latent  = pd.read_table(args.spotLatent, header = None, index_col = None).values
	spot_latent  = torch.FloatTensor(preprocess_features(spot_latent))
	print(spot_latent.size())

	# Spot-gene neighbors
	print("Spot-gene neighbors")
	adj_gene     = pd.read_table(args.spotGene, header = None, index_col = None).values
	nei_gene_n   = torch.LongTensor(adj_gene)
	if args.GeneLatent is None:
		Gene_latent  = sp.eye(args.Node_list[1])
		Gene_latent  = torch.FloatTensor(preprocess_features(Gene_latent).todense())
	else:
		Gene_latent  = pd.read_csv(args.GeneLatent, header = 0, index_col = 0).values
		Gene_latent  = torch.FloatTensor(preprocess_features(Gene_latent))

	print(Gene_latent.size())

	#Spot-group 
	print("Spot-group")
	nei_group    = pd.read_table(args.spotGroup, header = None, index_col = None).values
	nei_group    = [torch.LongTensor(i) for i in nei_group]
	Group_laten  = sp.eye(args.Node_list[2])
	Group_laten  = torch.FloatTensor(preprocess_features(Group_laten).todense())

	print(Group_laten.size())

	####sematic-path
	#RNA feature for adjancy
	print("RNA feature for adjancy")
	spot_rnas    = pd.read_table(args.spotLatent, header = None, index_col = None).values
	dist_rna     = pairwise_distances(spot_rnas, metric = args.rnaMeasure)
	row_index    = []
	col_index    = []

	sorted_knn   = np.argsort(-dist_rna)

	for index in list(range( np.shape(dist_rna)[0] )):
		col_index.extend( sorted_knn[index, :args.knn].tolist() )
		row_index.extend( [index] * args.knn )

	adj_rna    = sp.coo_matrix( (np.ones( len(row_index) ), (row_index, col_index) ), 
								shape=( np.shape(dist_rna)[0], np.shape(dist_rna)[0] ), dtype=np.float32 )

	adj_rna    = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_rna))

	#visual feature for adjancy
	print("visual feature for adjancy")
	image_lat    = pd.read_table(args.visualFeature, header = None, index_col = None)
	dist_vis     = pairwise_distances(image_lat.values, metric = args.visMeasure)
	row_index    = []
	col_index    = []

	sorted_knn   = np.argsort(-dist_vis)

	for index in list(range( np.shape(dist_vis)[0] )):
		col_index.extend( sorted_knn[index, :args.knn].tolist() )
		row_index.extend( [index] * args.knn )

	adj_vis    = sp.coo_matrix( (np.ones( len(row_index) ), (row_index, col_index) ), 
								shape=( np.shape(dist_vis)[0], np.shape(dist_vis)[0] ), dtype=np.float32 )

	adj_vis    = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_vis))
	
	#spot location for adjancy
	print("spot location for adjancy")
	spot_loc     = pd.read_table(args.spatialLocation, header = None, index_col = None)
	dist_loc     = pairwise_distances(spot_loc.values, metric = args.locMeasure)

	row_index    = []
	col_index    = []

	sorted_knn   = np.argsort(dist_loc)

	for index in list(range( np.shape(dist_loc)[0] )):
		col_index.extend( sorted_knn[index, :args.knn].tolist() )
		row_index.extend( [index] * args.knn )

	adj_loc    = sp.coo_matrix( (np.ones( len(row_index) ), (row_index, col_index) ), 
								shape=( np.shape(dist_loc)[0], np.shape(dist_loc)[0] ), dtype=np.float32 )

	adj_loc    = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_loc))

	# reCalculate for at least three semantic paths
	pos   = pd.read_table(args.pos_pair, header = None, index_col = None).values
	pos   = torch.FloatTensor(pos)

	#return [nei_gene_n, nei_group], [spot_latent, Gene_latent, Group_laten], [adj_rna, adj_vis, adj_loc, adj_group], cci_spots_n, pos
	return [nei_gene_n, nei_group], [spot_latent, Gene_latent, Group_laten], [adj_rna, adj_vis, adj_loc], None, pos

def load_data_general( args, pattern = "RNA_vis" ):
	# The order of node types: cell, gene, group
	#Spot-RNA
	print("Spot-RNA")
	spot_latent  = pd.read_table(args.spotLatent, header = None, index_col = None).values
	spot_latent  = torch.FloatTensor(preprocess_features(spot_latent))

	# Spot-gene neighbors
	print("Spot-gene neighbors")
	adj_gene     = pd.read_table(args.spotGene, header = None, index_col = None).values
	nei_gene_n   = torch.LongTensor(adj_gene)
	if args.GeneLatent is None:
		Gene_latent  = sp.eye(args.Node_list[1])
		Gene_latent  = torch.FloatTensor(preprocess_features(Gene_latent).todense())
	else:
		Gene_latent  = pd.read_csv(args.GeneLatent, header = 0, index_col = 0).values
		Gene_latent  = torch.FloatTensor(preprocess_features(Gene_latent))

	#Spot-group 
	print("Spot-group")
	nei_group    = pd.read_table(args.spotGroup, header = None, index_col = None).values
	nei_group    = [torch.LongTensor(i) for i in nei_group]
	Group_laten  = sp.eye(args.Node_list[2])
	Group_laten  = torch.FloatTensor(preprocess_features(Group_laten).todense())

	####sematic-path
	print("RNA feature for adjancy")
	spot_rnas    = pd.read_table(args.spotLatent, header = None, index_col = None).values
	dist_rna     = pairwise_distances(spot_rnas, metric = args.rnaMeasure)
	row_index    = []
	col_index    = []

	sorted_knn   = np.argsort(-dist_rna)

	for index in list(range( np.shape(dist_rna)[0] )):
		col_index.extend( sorted_knn[index, :args.knn].tolist() )
		row_index.extend( [index] * args.knn )

	adj_rna    = sp.coo_matrix( (np.ones( len(row_index) ), (row_index, col_index) ), 
								shape=( np.shape(dist_rna)[0], np.shape(dist_rna)[0] ), dtype=np.float32 )

	adj_rna    = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_rna))

	#visual feature for adjancy
	print("visual feature for adjancy")
	image_lat    = pd.read_table(args.visualFeature, header = None, index_col = None)
	dist_vis     = pairwise_distances(image_lat.values, metric = args.visMeasure)
	row_index    = []
	col_index    = []

	sorted_knn   = np.argsort(-dist_vis)

	for index in list(range( np.shape(dist_vis)[0] )):
		col_index.extend( sorted_knn[index, :args.knn].tolist() )
		row_index.extend( [index] * args.knn )

	adj_vis    = sp.coo_matrix( (np.ones( len(row_index) ), (row_index, col_index) ), 
								shape=( np.shape(dist_vis)[0], np.shape(dist_vis)[0] ), dtype=np.float32 )

	adj_vis    = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_vis))
	
	#spot location for adjancy
	print("spot location for adjancy")
	spot_loc     = pd.read_table(args.spatialLocation, header = None, index_col = None)
	dist_loc     = pairwise_distances(spot_loc.values, metric = args.locMeasure)

	row_index    = []
	col_index    = []

	sorted_knn   = dist_loc.argsort(axis=1)

	for index in list(range( np.shape(dist_loc)[0] )):
		col_index.extend( sorted_knn[index, :args.knn].tolist() )
		row_index.extend( [index] * args.knn )

	adj_loc    = sp.coo_matrix( (np.ones( len(row_index) ), (row_index, col_index) ), 
								shape=( np.shape(dist_loc)[0], np.shape(dist_loc)[0] ), dtype=np.float32 )

	adj_loc    = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_loc))

	#spot-spot similarity within one group
	print("spot-spot similarity by cell-cell communication")
	#cci_spots    = pd.read_table(args.cci_files, header = None, index_col = None).values
	#cci_spots_n  = torch.LongTensor(cci_spots)

	# reCalculate for at least three semantic paths
	pos   = pd.read_table(args.pos_pair, header = None, index_col = None).values
	pos   = torch.FloatTensor(pos)

	adj_groupss = None

	if pattern == "RNA":
		adj_groupss = [adj_rna]
	elif pattern == "vis":
		adj_groupss = [adj_vis]
	elif pattern == "loc":
		adj_groupss = [adj_loc]
	elif pattern == "RNA_vis":
		adj_groupss = [adj_rna, adj_vis]
	elif pattern == "RNA_loc":
		adj_groupss = [adj_rna, adj_loc]
	else:
		adj_groupss = [adj_vis, adj_loc]

	return [nei_gene_n, nei_group], [spot_latent, Gene_latent, Group_laten], adj_groupss, None, pos

def load_data_RNA( args ):
	# The order of node types: cell, gene, group
	#RNA
	print("RNA")
	Gene_latent  = sp.eye(args.Node_list[0])
	Gene_latent  = torch.FloatTensor(preprocess_features(Gene_latent).todense())

	# Spot-gene neighbors
	print("gene-spot neighbors")
	adj_spot     = pd.read_table(args.geneSpot, header = None, index_col = None)
	adj_spot_n   = torch.LongTensor(adj_spot.values)

	spot_latent  = pd.read_table(args.spotLatent, header = None, index_col = None).values
	spot_latent  = torch.FloatTensor(preprocess_features(spot_latent))

	#Spot-group 
	print("gene-group")
	nei_group    = pd.read_table(args.geneGroup, header = None, index_col = None).values
	nei_group    = [torch.LongTensor(i) for i in nei_group]

	Group_laten  = sp.eye(args.Node_list[2])
	Group_laten  = torch.FloatTensor(preprocess_features(Group_laten).todense())


	# reCalculate for at least three semantic paths
	pos   = pd.read_table(args.pos_pair_gene, header = None, index_col = None).values
	pos   = torch.FloatTensor(pos)

	return [adj_spot_n, nei_group], [Gene_latent, spot_latent, Group_laten], None, pos


def read_dataset( File1 = None, File2 = None,  transpose = True, test_size_prop = 0.15, state = 0 ):

	### File1 for raw reads count 
	if File1 is not None:
		adata = sc.read(File1)

		if transpose:
			adata = adata.transpose()
	else:
		adata = None
	
	### File2 for cell group information
	label_ground_truth = []

	if state == 0 :

		if File2 is not None:

			Data2 = pd.read_csv( File2, header=0, index_col=0 )
			## preprocessing for latter evaluation

			group = Data2['Group'].values

			for g in group:
				g = int(g.split('Group')[1])
				label_ground_truth.append(g)

		else:
			label_ground_truth =  np.ones( len( adata.obs_names ) )

	if test_size_prop > 0 :
		train_idx, test_idx = train_test_split(np.arange(adata.n_obs), 
											   test_size = test_size_prop, 
											   random_state = 200)
		spl = pd.Series(['train'] * adata.n_obs)
		spl.iloc[test_idx]  = 'test'
		adata.obs['split']  = spl.values
		
	else:
		train_idx, test_idx = list(range( adata.n_obs )), list(range( adata.n_obs ))

		spl = pd.Series(['train'] * adata.n_obs)
		adata.obs['split']       = spl.values
		
	adata.obs['split'] = adata.obs['split'].astype('category')
	adata.obs['Group'] = label_ground_truth
	adata.obs['Group'] = adata.obs['Group'].astype('category')
	
	print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))
	
	### here, adata with cells * features
	return adata, train_idx, test_idx, label_ground_truth


def normalize( adata, filter_min_counts=True, size_factors=True, 
			   normalize_input=False, logtrans_input=True):

	if filter_min_counts:
		sc.pp.filter_genes(adata, min_counts=1)
		sc.pp.filter_cells(adata, min_counts=1)

	if size_factors or normalize_input or logtrans_input:
		adata.raw = adata.copy()
	else:
		adata.raw = adata

	if logtrans_input:
		sc.pp.log1p(adata)

	if size_factors:
		#adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
		adata.obs['size_factors'] = np.log( np.sum( adata.X, axis = 1 ) )
	else:
		adata.obs['size_factors'] = 1.0

	if normalize_input:
		sc.pp.scale(adata)

	return adata


def save_checkpoint(model, folder='./saved_model/', filename='model_best.pth.tar'):
	if not os.path.isdir(folder):
		os.mkdir(folder)

	torch.save(model.state_dict(), os.path.join(folder, filename))

def load_checkpoint(file_path, model, use_cuda=False):

	if use_cuda:
		device = torch.device( "cuda" )
		model.load_state_dict( torch.load(file_path) )
		model.to(device)
		
	else:
		device = torch.device('cpu')
		model.load_state_dict( torch.load(file_path, map_location=device) )

	model.eval()
	return model