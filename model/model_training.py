# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:40:25 2022

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
from tqdm import tqdm
from utilities import normalize, read_dataset, load_data_RNA
from utilities import load_data_naive, load_data_general, load_cci_ligand_receptors
from module import Cell_module, Mp_encoder_HIN, Sc_encoder_HIN, CCC
from AE_model import AE


def Trian_hin_model( args, model = "Cell", pattern = "RNA_vis" ):
	if model == "Cell":
		if pattern is not None:
			nei_index, features, sematic_path, cci_Path, positive_pairs = load_data_general(args, pattern)
		else:
			nei_index, features, sematic_path, cci_Path, positive_pairs = load_data_naive(args)
	else:
		nei_index, features, sematic_path, positive_pairs = load_data_RNA(args)

	feat_dim_list  = [i.shape[1] for i in features]
	Path           = int(len(sematic_path))
	model = Cell_module(args.hidden_dim, feat_dim_list, args.feat_drop, args.attn_drop,
				        Path, args.sample_rate, (args.Node_type-1), args.tau, args.lam)
	optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
	
	print('Start model training')
	if args.use_cuda:
		model.cuda()
		feats   = [feat.cuda() for feat in features]
		mps     = [mp.cuda() for mp in sematic_path]
		pos     = positive_pairs.cuda()

	cnt_wait  = 0
	best      = 1e9
	best_t    = 0
	starttime = datetime.datetime.now()

	train_loss_list = []

	for epoch in range(args.max_training):
		model.train()
		optim.zero_grad()
		#print(str(epoch))
		lori_mp, lori_sc, lori_mp2mp, lori_sc2sc = model(feats, pos, mps, nei_index)

		loss  = args.lam * (lori_mp+lori_mp2mp) + (1 - args.lam) * (lori_sc+lori_sc2sc)

		if epoch %10==0 :
			print( str(epoch) + " mp: " + str((lori_mp+lori_mp2mp).data.cpu())+"  sc: "+ str((lori_sc+lori_sc2sc).data.cpu())+"  total: "+ str(loss.data.cpu()))
	
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
	embeds_mp = model.get_mp_embeds(feats, mps)
	embeds_sc = model.get_sc_embeds(feats, nei_index)

	name_pre = '{}_{}_{}_{}_{}_{}_{}_{}'.format( args.lr, args.patience, args.tau, args.lam, args.feat_drop, args.attn_drop, args.sample_rate[0],epoch)
	torch.save(model.state_dict(), args.outPath + '{}_best.pkl'.format(name_pre))

	atten_mul_V, atten_mul_R, atten_recep, atten_adj_V, atten_adj_L = model.get_attention_mp(feats, mps)    
	atten_var, atten_inter  = model.get_attention_sc(feats, nei_index)
	
	if args.save_emb:
		embeds_mp1      =  pd.DataFrame(data=embeds_mp.data.cpu().numpy()).to_csv( args.outPath + 'Spot_rep_mp_{}.csv'.format(name_pre) )
		embeds_sc1      =  pd.DataFrame(data=embeds_sc.data.cpu().numpy()).to_csv( args.outPath + 'Spot_rep_sc_{}.csv'.format(name_pre) )
		atten_mul_V1    =  pd.DataFrame(data=atten_mul_V.data.cpu().numpy()).to_csv( args.outPath + 'Atten_SC_mul_view_{}.csv'.format(name_pre) )
		atten_var1      =  pd.DataFrame(data=atten_var.data.cpu().numpy()).to_csv( args.outPath + 'Atten_MP_gene_{}.csv'.format(name_pre) )
		atten_inter1    =  pd.DataFrame(data=atten_inter.data.cpu().numpy()).to_csv( args.outPath + 'Atten_MP_inter_{}.csv'.format(name_pre) )

	return loss.data.cpu().numpy(), epoch


def Trian_single_HIN_model( args, model = "Cell", pattern = "SC" ):

	if model == "Cell":
		nei_index, features, sematic_path, cci_Path, positive_pairs = load_data_naive(args)
	else:
		nei_index, features, sematic_path, positive_pairs = load_data_RNA(args)

	feat_dim_list  = [i.shape[1] for i in features]
	print(feat_dim_list)
	
	if pattern == "MP":
		Path  = int(len(sematic_path))
		model = Mp_encoder_HIN(args.hidden_dim, feat_dim_list, args.feat_drop, args.attn_drop, Path, args.tau)
		if args.use_cuda:
			mps = [mp.cuda() for mp in sematic_path]
		
	else:
		model = Sc_encoder_HIN(args.hidden_dim, feat_dim_list, args.feat_drop, args.attn_drop,
							   args.sample_rate, (args.Node_type-1), args.tau)
		mps   = None

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

	for epoch in range(args.max_training):
		model.train()
		optim.zero_grad()
		loss = model(feats, pos, mps, nei_index)
		#loss = args.lam * loss

		if epoch %10 == 0 :
			print( str(epoch) + "  " + pattern + ": " + str(loss.data.cpu()))
	
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

	if pattern == "MP":
		embeds = model.get_mp_embeds(feats, mps)
	else:
		embeds = model.get_sc_embeds(feats, nei_index)

	name_pre = '{}_{}_{}_{}_{}_{}_{}_{}'.format( args.lr, args.patience, args.tau, args.lam, args.feat_drop, args.attn_drop, args.sample_rate[0],epoch)
	torch.save(model.state_dict(), args.outPath + '{}_best.pkl'.format(name_pre))
	
	if args.save_emb:
		embeds1      =  pd.DataFrame(data=embeds.data.cpu().numpy()).to_csv( args.outPath + 'Spot_rep_{}_{}.csv'.format(pattern, name_pre) )


def Trian_CCI_contrast( args ):

	nei_adj, spots_ligand, spots_recep, pos = load_cci_ligand_receptors(args)

	args.cci_pairs = spots_ligand.size(1)
	print('Size of CCI pairs: ' + str(args.cci_pairs))
	
	model = CCC(args.cci_pairs, 1, args.attn_drop, [args.cci_pairs, 100], args.tau)
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

	for epoch in range(args.max_training):
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
			if (abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2])  <= 0.003 :
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

	name_pre = '{}_{}_{}_{}_{}_{}_{}_{}'.format( args.lr_cci, args.patience, args.tau, args.lam, args.feat_drop, args.attn_drop, args.sample_rate[0],epoch)
	torch.save(model.state_dict(), args.outPath + '{}_best.pkl'.format(name_pre))

	LR_activity = model.return_LR_value(nei_adj, spots_ligand, spots_recep)
	
	if args.save_emb:
		LR_activity1 =  pd.DataFrame(data=LR_activity.data.cpu().numpy()).to_csv( args.outPath + 'Spot_LR_activity_{}.csv'.format(name_pre) )


def RNA_encoding_train(args, RNA_file = None, outDir = "results", sub_name = "AE_encoding"):

	args.batch_size_T   = 128
	args.epoch_per_test = 10

	adata, train_index, test_index, _ = read_dataset( File1 = RNA_file, transpose = True, test_size_prop = 0.1 )
	adata  = normalize( adata, filter_min_counts=True, size_factors=True,
						normalize_input=False, logtrans_input=True ) 
	
	_, Nfeature1    =  np.shape( adata.X )

	train           = data_utils.TensorDataset( torch.from_numpy( adata[train_index].X ),
												torch.from_numpy( adata.raw[train_index].X ), 
												torch.from_numpy( adata.obs['size_factors'][train_index].values ) )
	train_loader    = data_utils.DataLoader( train, batch_size = args.batch_size_T, shuffle = True )

	test            = data_utils.TensorDataset( torch.from_numpy( adata[test_index].X ),
												torch.from_numpy( adata.raw[test_index].X ), 
												torch.from_numpy( adata.obs['size_factors'][test_index].values ) )
	test_loader     = data_utils.DataLoader( test, batch_size = len(test_index), shuffle = False )

	total           = data_utils.TensorDataset( torch.from_numpy( adata.X ),
												torch.from_numpy( adata.obs['size_factors'].values ) )
	total_loader    = data_utils.DataLoader( total, batch_size = args.batch_size_T, shuffle = False )

	AE_structure = [Nfeature1, 1000, 50, 50, 1000, Nfeature1]
	print(Nfeature1)
	
	model        = AE( [Nfeature1, 1000, 50], layer_d = [50, 1000], 
					   hidden1 = 1000, args = args, droprate = 0, type = "NB"  )

	if args.use_cuda:
		model.cuda()

	model.fit( train_loader, test_loader )

	save_name_pre = '{}_{}_{}-{}'.format( sub_name, args.batch_size_T, args.lr_AET , '-'.join( map(str, AE_structure )) )
	latent_z      = model.predict(total_loader, out='z' )

	torch.save(model, outDir + '/{}_RNA_AE_model.pt'.format(save_name_pre) )
	latent_z1  = pd.DataFrame( latent_z, index= adata.obs_names ).to_csv( outDir + '/{}_RNA.csv'.format(save_name_pre) ) 
