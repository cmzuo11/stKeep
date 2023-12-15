# -*- coding: utf-8 -*-
"""

@author: chunman zuo
"""

import os, sys
import argparse
import time
import random
import datetime
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
import warnings

from torch.utils.data.dataloader import default_collate
from torch import optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utilities import normalize, load_data_RNA, load_data_cell, load_ccc_data
from modules import Cell_module, Gene_module, CCI_model, AE


def Trian_cell_model( args):
	nei_index, features, sematic_path, positive_pairs, cellName = load_data_cell(args)

	feat_dim_list  = [i.shape[1] for i in features]
	Path           = int(len(sematic_path))
	model = Cell_module(args.hidden_dim, feat_dim_list, args.feat_drop, args.attn_drop,
				        Path, args.sample_rate, (args.Node_type-1), args.tau, args.lam)
	optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
	
	print('Start model training')
	if args.use_cuda:
		model.cuda()
		feats   = [feat.cuda() for feat in features]
		smps    = [smp.cuda() for smp in sematic_path]
		pos     = positive_pairs.cuda()

	cnt_wait  = 0
	best      = 1e9
	best_t    = 0
	starttime = datetime.datetime.now()

	train_loss_list = []

	for epoch in range(500):
		model.train()
		optim.zero_grad()
		#print(str(epoch))
		lori_mp, lori_sc, lori_mp2mp, lori_sc2sc = model(feats, pos, smps, nei_index)

		loss  = args.lam * (lori_mp+lori_mp2mp) + (1 - args.lam) * (lori_sc+lori_sc2sc)

		if epoch %10==0 :
			print( str(epoch) + " Semantic: " + str((lori_mp+lori_mp2mp).data.cpu())+"  Hierarchical: "+ str((lori_sc+lori_sc2sc).data.cpu())+"  total: "+ str(loss.data.cpu()))
	
		if loss < best:
			best     = loss
			best_t   = epoch
			cnt_wait = 0
		else:
			cnt_wait += 1

		if cnt_wait == args.patience:
			print(  str(best_t) + ' Early stopping!' )
			break

		loss.backward()
		optim.step()
	
	model.eval()
	endtime   = datetime.datetime.now()
	time      = (endtime - starttime).seconds
	print("Total time: ", time, "s")
	embeds_mp = model.get_mp_embeds(feats, smps)
	embeds_sc = model.get_sc_embeds(feats, nei_index)

	#torch.save(model.state_dict(), args.outPath + 'Cell_module.pkl' )

	atten_mul_V, atten_mul_R, atten_recep, atten_adj_V, atten_adj_L = model.get_attention_mp(feats, smps)    
	atten_var, atten_inter  = model.get_attention_sc(feats, nei_index)

	pd.DataFrame(data=embeds_mp.data.cpu().numpy(), index= cellName).to_csv( args.outPath + 'Semantic_representations.txt' , sep='\t')
	pd.DataFrame(data=embeds_sc.data.cpu().numpy(), index= cellName).to_csv( args.outPath + 'Hierarchical_representations.txt' , sep='\t')


def Trian_gene_model( args ):

	nei_index, features, positive_pairs, geneSymbol = load_data_RNA(args)
	feat_dim_list  = [i.shape[1] for i in features]
	
	model = Gene_module(args.hidden_dim, feat_dim_list, args.feat_drop, args.attn_drop,
						args.sample_rate, (args.Node_type-1), args.tau)

	optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
	
	print('Start model training')
	if args.use_cuda:
		model.cuda()
		feats   = [feat.cuda() for feat in features]
		pos     = positive_pairs.cuda()

	cnt_wait  = 0
	best      = 1e9
	best_t    = 0
	starttime = datetime.datetime.now()

	train_loss_list = []

	for epoch in range(200):
		model.train()
		optim.zero_grad()
		loss = model(feats, pos, nei_index)

		if epoch %10 == 0 :
			print( str(epoch) + "  " + ": " + str(loss.data.cpu()))
	
		if loss < best:
			best     = loss
			best_t   = epoch
			cnt_wait = 0
		else:
			cnt_wait += 1

		if cnt_wait == args.patience:
			print(  str(best_t) + ' Early stopping!' )
			break

		loss.backward()
		optim.step()
	
	model.eval()
	endtime   = datetime.datetime.now()
	time      = (endtime - starttime).seconds
	print("Total time: ", time, "s")

	embeds  = model.get_embeds(feats, nei_index)
	pd.DataFrame( data=embeds.data.cpu().numpy(), index = geneSymbol.tolist() ).to_csv( args.outPath + 'Gene_module_representation.txt' , sep='\t') 
	#torch.save( model.state_dict(), args.outPath + 'Gene_model.pkl' )


def Trian_CCC_model( args):

	nei_adj, spots_ligand, spots_recep, pos, cellName, LRP_name = load_ccc_data(args)

	args.cci_pairs = spots_ligand.size(1)
	print('Size of CCC pairs: ' + str(args.cci_pairs))
	
	model = CCI_model(args.cci_pairs, 1, args.attn_drop, [args.cci_pairs, 100], args.tau)
	optim = torch.optim.Adam(model.parameters(), lr=args.lr_cci, weight_decay=args.l2_coef)
	
	print('Start model training')
	if args.use_cuda:
		model.cuda()
		nei_adj      = nei_adj.cuda()
		spots_ligand = spots_ligand.cuda()
		spots_recep  = spots_recep.cuda()
		pos          = pos.cuda()

	cnt_wait  = 0
	best      = 1e9
	best_t    = 0
	rela_loss = 1000
	starttime = datetime.datetime.now()

	train_loss_list = []

	for epoch in range(1000):
		model.train()
		optim.zero_grad()

		cost = model(nei_adj, spots_ligand, spots_recep, pos)
		cost = cost*100

		train_loss_list.append( cost  )

		if epoch %10==0 :
			if len(train_loss_list) >= 2 :
				print( str(epoch) + " cost: " + str(cost.data.cpu()) + " " + str(abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2]) )
			else:
				print( str(epoch) + " cost: " + str(cost.data.cpu()) )

		if (epoch>50) and (len(train_loss_list) >= 2) :
			if (abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2])  <= 0.005:
				print( abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] )
				print( str(train_loss_list[-1])+ " " + str(train_loss_list[-2]) + " converged!!!" )
				print( epoch )
				break

		cost.backward()
		optim.step()

	model.eval()
	endtime   = datetime.datetime.now()
	time      = (endtime - starttime).seconds
	print("Total time: ", time, "s")

	#torch.save(model.state_dict(), args.outPath + 'CCC_module.pkl')

	LR_activity  = model.return_LRP_strength(nei_adj, spots_ligand, spots_recep)

	pd.DataFrame(data=LR_activity.data.cpu().numpy(), index = cellName.tolist(), columns = LRP_name.tolist() ).to_csv( args.outPath + 'CCC_module_LRP_strength.txt', sep='\t')


def RNA_encoding_train(args, adata = None, test_size_prop = 0.1):

	args.batch_size_T   = 128
	args.epoch_per_test = 10

	if test_size_prop > 0 :
		train_index, test_index = train_test_split(np.arange(adata.n_obs), 
												   test_size    = test_size_prop, 
												   random_state = 200)
	else:
		train_index, test_index = list(range( adata.n_obs )), list(range( adata.n_obs ))
			
	adata  = normalize( adata, filter_min_counts=True, size_factors=True,
						normalize_input=False, logtrans_input=True ) 
	
	Nsample1, Nfeature1 =  np.shape( adata.X )

	train           = data_utils.TensorDataset( torch.from_numpy( sp.csr_matrix.toarray(adata[train_index].X) ),
												torch.from_numpy( sp.csr_matrix.toarray(adata.raw[train_index].X) ), 
												torch.from_numpy( adata.obs['size_factors'][train_index].values ) )
	train_loader    = data_utils.DataLoader( train, batch_size = args.batch_size_T, shuffle = True )

	test            = data_utils.TensorDataset( torch.from_numpy( sp.csr_matrix.toarray(adata[test_index].X) ),
												torch.from_numpy( sp.csr_matrix.toarray(adata.raw[test_index].X) ), 
												torch.from_numpy( adata.obs['size_factors'][test_index].values ) )
	test_loader     = data_utils.DataLoader( test, batch_size = len(test_index), shuffle = False )

	total           = data_utils.TensorDataset( torch.from_numpy( sp.csr_matrix.toarray(adata.X) ),
												torch.from_numpy( adata.obs['size_factors'].values ) )
	total_loader    = data_utils.DataLoader( total, batch_size = args.batch_size_T, shuffle = False )


	AE_structure = [Nfeature1, 1000, 50, 1000, Nfeature1]
	model        = AE( [Nfeature1, 1000, 50], layer_d = [50, 1000], 
					   hidden1 = 1000, args = args, droprate = 0, type = "NB"  )

	if args.use_cuda:
		model.cuda()

	model.fit( train_loader, test_loader )

	latent_z   = model.predict(total_loader, out='z' )

	#torch.save(model, args.outPath + '/Cell_encoding_AE_model.pt')
	pd.DataFrame( latent_z, index= adata.obs_names ).to_csv( args.outPath + '/Cell_encoding_AE.txt', sep='\t'  ) 

