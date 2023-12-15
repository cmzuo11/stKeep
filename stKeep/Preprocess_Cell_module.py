# -*- coding: utf-8 -*-
"""

@author: chunman zuo
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import time
import os
import torch
import random
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

from utilities import parameter_setting, get_cell_gene_neighbors, get_cell_positive_pairs, get_gene_modules_data, get_gene_pairs
from image_processing import tiling, train_simCLR_sImage, extract_representation_simCLR_model, extract_representation_resnet50_model
from model_training import RNA_encoding_train


def Preprocessing( args ):

	start = time.time()
	args.use_cuda       = args.use_cuda and torch.cuda.is_available()	
	args.tillingPath    = Path( args.inputPath + 'tmp/' )
	args.tillingPath.mkdir(parents=True, exist_ok=True)
	args.outPath        = args.inputPath + 'stKeep/'
	Path(args.outPath).mkdir(parents=True, exist_ok=True)

	print('1---load spatial transcriptomics data')
	adata      = sc.read_visium( args.inputPath )
	adata.var_names_make_unique()
	print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

	library_id = list(adata.uns["spatial"].keys())[0]
	scale      = adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]
	image_coor = adata.obsm["spatial"] * scale
	adata.obs["imagecol"] = image_coor[:, 0]
	adata.obs["imagerow"] = image_coor[:, 1]
	adata.uns["spatial"][library_id]["use_quality"] = 'hires'

	#repeat our results by the following code
	annota_clu = pd.read_table(args.inputPath + args.annoFile, header=0, index_col=0)
	adata.obs['Annotation']  =  annota_clu.loc[adata.obs_names].values[:,0].tolist()
	remain_int = np.where(adata.obs['Annotation'].values.astype('str')!='nan')[0]
	hvgs       = pd.read_table(args.inputPath + '151507_repeat_gene.txt', header=0, index_col=0)
	res        = [adata.var_names.tolist().index(item) for item in hvgs.index.tolist() if item in adata.var_names.tolist() ]
	adata2     = adata[remain_int, res]

	print('2---save spatial location')
	spot_loc = { 'imagerow': adata2.obs['imagerow'].values.tolist(), 'imagecol': adata2.obs['imagecol'].values.tolist() }
	pd.DataFrame(spot_loc, index = adata2.obs_names.tolist()).to_csv( args.outPath + args.spatialLocation, sep='\t' )

	print('3---calculate spot-group adjance')
	args.class_mapping = {label: idx for idx, label in enumerate(np.unique(adata2.obs['Annotation'].values.astype('str')))}
	adata2.obs['classlabel'] = adata2.obs['Annotation'].map(args.class_mapping)
	pd.DataFrame(adata2.obs['classlabel']).to_csv( args.outPath + args.spotGroup, sep='\t' )


	print('4---calculate cell positive pairs')
	get_cell_positive_pairs(adata2, args)
	

	print('5---calculate cell encoding')
	RNA_encoding_train(args, adata2 )
	

	print('6---save cell-gene neighborhoods')
	get_cell_gene_neighbors(adata2, args)


	print('7---extracting histologicald features from H&E images')
	tiling(adata2, args.tillingPath, target_size = args.image_size)
	if args.Hismodel == 'SimCLR':
		train_simCLR_sImage( args, args.outPath )
		extract_representation_simCLR_model( args, adata2 ) 
	else:
		extract_representation_resnet50_model(args, adata2)

	#df_vis = pd.read_csv(args.inputPath + 'stKeep/128_0.5_200_128_simCLR_reprensentation.csv', header=0, index_col=0)
	#df_vis.loc[adata2.obs_names, ].to_csv( args.outPath + args.visualFeature , sep='\t')

	duration = time.time() - start
	print('Finish training, total time is: ' + str(duration) + 's' )

if __name__ == "__main__":

	parser  =  parameter_setting()
	args    =  parser.parse_args()
	Preprocessing(args)
