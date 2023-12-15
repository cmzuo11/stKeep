# -*- coding: utf-8 -*-
"""

@author: chunman zuo
"""
import numpy
import warnings
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import random
from pathlib import Path

from utilities import parameter_setting
from model_training import Trian_CCC_model


def train_with_argas( args ):

	## random seed 
	numpy.random.seed( args.seed )
	random.seed( args.seed )
	torch.manual_seed( args.seed )
	torch.cuda.manual_seed( args.seed )

	start = time.time()
	args.use_cuda        = args.use_cuda and torch.cuda.is_available()

	args.outPath         = args.inputPath + 'stKeep/'	
	args.spatialLocation = args.outPath + args.spatialLocation
	args.pos_pair        = args.outPath + args.pos_pair 

	args.Ligands_exp     = args.outPath + args.Ligands_exp
	args.Receptors_exp   = args.outPath + args.Receptors_exp

	args.patience        = 15
	args.lr_cci          = 0.001
	args.attn_drop       = 0
	args.tau             = 0.05

	Trian_CCC_model(args)
	

	duration = time.time() - start
	print('Finish training, total time is: ' + str(duration) + 's' )

if __name__ == "__main__":
	parser  =  parameter_setting()
	args    =  parser.parse_args()
	train_with_argas(args)
