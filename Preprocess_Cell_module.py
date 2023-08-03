# -*- coding: utf-8 -*-
"""

@author: chunman zuo
"""
import stlearn as st
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import time
import os
import torch
import random
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

from stKeep.utilities import parameter_setting, get_cell_gene_neighbors, get_cell_positive_pairs, get_gene_modules_data, get_gene_pairs
from stKeep.image_processing import tiling, train_simCLR_sImage, extract_representation_simCLR_model
from stKeep.model_training import RNA_encoding_train
st.settings.set_figure_params(dpi=180)


def Preprocessing( args ):

	start = time.time()
	args.use_cuda       = args.use_cuda and torch.cuda.is_available()	
	args.tillingPath    = Path( args.inputPath + 'tmp/' )
	args.tillingPath.mkdir(parents=True, exist_ok=True)
	args.outPath        = args.inputPath + 'stKeep/'
	Path(args.outPath).mkdir(parents=True, exist_ok=True)

	print('1---load spatial transcriptomics and histological data')
	adata      = st.Read10X( Path(args.inputPath) )
	print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

	#repeat our results by the following code
	annota_clu = pd.read_table(args.inputPath + args.annoFile, header=0, index_col=0)
	adata.obs['Annotation']  =  annota_clu.loc[adata.obs_names].values[:,0].tolist()
	remain_int = np.where(adata.obs['Annotation'].values.astype('str')!='nan')[0]
	hvgs       = pd.read_table(args.inputPath + '151507_Exp_2000_hvg_counts.txt', header=0, index_col=0)
	res        = [adata.var_names.tolist().index(item) for item in hvgs.index.tolist() if item in adata.var_names.tolist() ]
	adata2     = adata[remain_int, res]


	print('2---saving spatial location data')
	#adata2   = st.convert_scanpy(adata2)
	spot_loc = { 'imagerow': adata2.obs['imagerow'].values.tolist(), 'imagecol': adata2.obs['imagecol'].values.tolist() }
	pd.DataFrame(spot_loc, index = adata2.obs_names.tolist()).to_csv( args.outPath + args.spatialLocation, sep='\t' )


	print('3---saving spot-group adjance into  Spot_groups.txt')
	args.class_mapping = {label: idx for idx, label in enumerate(np.unique(adata2.obs['Annotation'].values.astype('str')))}
	adata2.obs['classlabel'] = adata2.obs['Annotation'].map(args.class_mapping)
	pd.DataFrame(adata2.obs['classlabel']).to_csv( args.outPath + args.spotGroup, sep='\t' )


	print('4---saving cell positive pairs into Spot_positive_pairs.txt')
	get_cell_positive_pairs(adata2, args)
	

	print('5---Start training autoencoder-based framework for learning latent features')
	RNA_encoding_train(args, adata2 )
	

	print('6---saving cell-gene neighborhoods into Spot_gene_neighbors.txt')
	get_cell_gene_neighbors(adata2, args)


	print('7---tilling histologicald data and train sinCLR model')
	tiling(adata2, args.tillingPath, target_size = args.image_size)
	train_simCLR_sImage( args, args.outPath )
	extract_representation_simCLR_model( args, adata2 ) 

	#df_vis = pd.read_csv(args.inputPath + 'CMSSL_new/128_0.5_200_128_simCLR_reprensentation.csv', header=0, index_col=0)
	#df_vis.loc[adata2.obs_names, ].to_csv( args.outPath + args.visualFeature , sep='\t')

	duration = time.time() - start
	print('Finish training, total time is: ' + str(duration) + 's' )

if __name__ == "__main__":

	parser  =  parameter_setting()
	args    =  parser.parse_args()
	Preprocessing(args)
