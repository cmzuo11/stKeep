#!/usr/bin/env python
"""

# Author: Chunman zuo
# File Name: __init__.py
# Description:
"""


from .model_training import Train_cell_model, Train_gene_model, Train_CCC_model, RNA_encoding_train

from .utilities import parameter_setting, get_cell_gene_neighbors, get_cell_positive_pairs, get_gene_modules_data, get_gene_pairs
from .image_processing import tiling, train_simCLR_sImage, extract_representation_simCLR_model, extract_representation_resnet50_model

from .utilities import normalize, load_data_RNA, load_data_cell, load_ccc_data, adjust_learning_rate
from .modules import Cell_module, Gene_module, CCI_model, AE, log_nb_positive, mse_loss

from .Layers import Hierarchical_encoder, Semantic_encoder, Contrast, LRP_attention, build_multi_layers, Contrast_single, Decoder_logNorm_NB, Decoder



