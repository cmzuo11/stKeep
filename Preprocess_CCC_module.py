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

from stKeep.utilities import parameter_setting, get_cell_gene_neighbors, get_cell_positive_pairs
from stKeep.utilities import get_CCC_data, get_gene_modules_data, get_gene_pairs

from stKeep.image_processing import tiling, train_simCLR_sImage, extract_representation_simCLR_model
from stKeep.model_training import RNA_encoding_train


def Preprocessing( args ):

	start = time.time()
	args.use_cuda       = args.use_cuda and torch.cuda.is_available()
	args.inputPath      = '/sibcb2/chenluonanlab7/cmzuo/workPath/CMSSL/spatial_result/DLPFC/151507/'
	
	args.tillingPath    = Path( args.inputPath + 'tmp/' )
	args.tillingPath.mkdir(parents=True, exist_ok=True)
	args.outPath        = args.inputPath + 'stKeep/'
	Path(args.outPath).mkdir(parents=True, exist_ok=True)

	print('load spatial transcriptomics and histological data')
	adata      = sc.read_visium( args.inputPath )
	adata.var_names_make_unique()

	print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

	annota_clu = pd.read_table(args.inputPath + args.annoFile, header=0, index_col=0)
	adata.obs['Annotation']  =  annota_clu.loc[adata.obs_names].values[:,0].tolist()

	remain_int = np.where(adata.obs['Annotation'].values.astype('str')!='nan')[0]
	adata1     = adata[remain_int]

	latent_mp  = pd.read_table(args.outPath + 'Semantic_representations.txt', header=0, index_col=0)
	latent_sc  = pd.read_table(args.outPath + 'Hierarchical_representations.txt', header=0, index_col=0)

	res        = [ adata1.obs_names.tolist().index(item) for item in latent_sc.index.tolist()  if item in adata1.obs_names.tolist() ]
	latent     = np.concatenate((latent_mp.values, latent_sc.values), axis=1)

	get_CCC_data(adata1, latent[res, :], args)

	duration   = time.time() - start
	print('Finish training, total time is: ' + str(duration) + 's' )


if __name__ == "__main__":

	parser  =  parameter_setting()
	args    =  parser.parse_args()
	Preprocessing(args)
