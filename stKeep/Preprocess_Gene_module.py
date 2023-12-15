# -*- coding: utf-8 -*-
"""

@author: chunman zuo
"""
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

from utilities import parameter_setting, get_cell_gene_neighbors, get_cell_positive_pairs, get_gene_modules_data, get_gene_pairs
from image_processing import tiling, train_simCLR_sImage, extract_representation_simCLR_model
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
	#print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

	#repeat our results by the following code
	annota_clu = pd.read_table(args.inputPath + args.annoFile, header=0, index_col=0)
	adata.obs['Annotation']  =  annota_clu.loc[adata.obs_names].values[:,0].tolist()
	remain_int = np.where(adata.obs['Annotation'].values.astype('str')!='nan')[0]
	hvgs       = pd.read_table(args.inputPath + '151507_repeat_gene.txt', header=0, index_col=0)
	res        = [adata.var_names.tolist().index(item) for item in hvgs.index.tolist() if item in adata.var_names.tolist() ]
	adata2     = adata[remain_int, res]

	args.class_mapping = {label: idx for idx, label in enumerate(np.unique(adata2.obs['Annotation'].values.astype('str')))}
	adata2.obs['classlabel'] = adata2.obs['Annotation'].map(args.class_mapping)


	print('2---saving gene-module data')
	gene_select = get_gene_pairs(adata2, args)
	get_gene_modules_data(adata2, args, gene_select)


	duration = time.time() - start
	print('Finish training, total time is: ' + str(duration) + 's' )

if __name__ == "__main__":

	parser  =  parameter_setting()
	args    =  parser.parse_args()
	Preprocessing(args)
