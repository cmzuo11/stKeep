# Dissecting tumor microenvironment from spatially resolved transcriptomics data by heterogeneous graph learning.

![image](https://github.com/cmzuo11/stKeep/blob/main/utilities/main-framework.png)

Overview of the stKeep model. a-c Given each SRT data with four-layer profiles: histological images (I), spatial locations (S), gene expression (X), histological regions (Y), and gene-gene interactions such as PPI, GRN, and LRP as the input, stKeep integrates them to construct HG for dissecting tumor ecosystems. d Cell module adopts a cell-centered HG to capture local hierarchical representations (〖R_i〗^1) through aggregating features from genes and regions by attention, while leveraging intercellular graphs including SLG, HSG, and TSG to learn global semantic representations (〖R_i〗^2), and collaboratively integrates two representations by self-supervised learning. e Gene module utilizes a gene-centered HG to learn low-dimensional representations by combining features from cells and clusters using attention, while ensuring co-relational gene pairs are embedded adjacent to each other using contrastive learning. f Cell-cell communication module leverages attention-based heterogeneous graphs to infer ligand-receptor interaction strength (H_i) for each cell by aggregating ligand information from the neighbors for a central cell, while guaranteeing that CCC patterns can characterize diverse cell-states within TME. Note that each graph indicates one LRP. g The unified framework with three modules (d-f) can be used to dissect tumor ecosystems by detecting spatial clusters and visualizing them, identifying cell-state-specific gene-modules and receptor-gene interaction networks, and inferring cellular communication strength.


# Installation

## Install stKeep

Installation was tested on Red Hat 7.6 with Python 3.8.18 and torch 1.13.0 on a machine with one 40-core Intel(R) Xeon(R) Gold 5115 CPU addressing with 132GB RAM, and two NVIDIA TITAN V GPUs addressing 24GB. stKeep is implemented in the Pytorch framework. Please run stKeep on CUDA if possible. 

### Install stKeep in the virtual environment by conda

#### Step 1: install conda: https://docs.anaconda.com/anaconda/install/index.html, and then create a envs named stKeep with python 3.8.18

```
conda create -n stKeep python=3.8.18 pip
conda activate stKeep
```

#### Step 2: automatically install stKeep from pypi website: https://pypi.org/project/stKeep/

```
pip install stKeep
cd stKeep
```

#### or you can install it from Github:

```
git clone https://github.com/cmzuo11/stKeep.git
cd stKeep
pip install .
```

## Install R packages 

* Install tested on R =4.0.0
* Install package 'Seurat' based on the Github https://github.com/satijalab/seurat
* install.packages("ggplot2")


# Quick start

## Input

* A general output of 10X pipeline, for example, a directory includes a file named filtered_feature_bc_matrix.h5, a directory named spatial with five files: tissue_positions_list.csv, tissue_lowres_image.png, tissue_hires_image.png, metrics_summary_csv.csv, scalefactors_json.json, and a directory named as filtered_feature_bc_matrix with three files: matrix.mtx.gz, features.tsv.gz, and barcodes.tsv.gz;  

* Take slice 151507 as an example, you can find it in the test_data.

  
## Run

### Step 1. Region annotation (for tumor samples)

You can adapt two different methods to define pathological regions, i.e., tumor regions:

#### Method 1 

* install the Loupe browser software from the website: https://www.10xgenomics.com/products/loupe-browser/downloads, and apply it to annotate cells based on tumor regions manually.

#### Method 2
  
* install the labelme software from the link on Github: https://github.com/wkentaro/labelme, and apply it to manually outline each tumor region based on our defined strategy, and save the annotation into a json file named 'tissue_hires_image.json' of a directory named image_segmentation.

* Define the classification for each spot based on above-generated json file. Here, we use IDC dataset as an example.

```
python ./stKeep/Image_cell_segmentation.py --inputPath ./test_data/IDC/ --jsonFile tissue_hires_image.json
```


### Step 2. Run cell module

#### Calculate the input data for the cell module

This function automatically calculates input data for the cell module, including the relations between cells/spots and genes, the links between cells/spots and annotated regions, 50-dimensional representations from highly variable genes, physical spatial location, 2048-dimensional visual features from histological data, and cell/spot positive pairs.

```
python ./stKeep/Preprocess_Cell_module.py --inputPath ./test_data/DLPFC_151507/
```
The running time mainly depends on the iteration of the histological image extraction model. It takes ~3h to generate the above-described files. You can modify the following parameters to reduce time:

* batch_size_I: defines the batch size for training histological image extraction model. The default value is 128. You can modify it based on your memory size. The larger the parameter, the less time.

* max_epoch_I: defines the max iteration for training histological image extraction model. The default value is 500. You can modify it. The smaller the parameter, the less time.

To reproduce the result, you should use the default parameters. 

Note: to reduce your waiting time, you can use the file 'Image_simCLR_reprensentation.txt' in the folder './DLPFC_151507/stKeep/'. If you can't find it, please download it from the link https://drive.google.com/drive/folders/1RTb_gHcpLhnbRMHrqn8tBtynesq5g5DI?usp=drive_link, and then put them into the './test_data/DLPFC_151507/stKeep/' folder.

Optionally, we have provided another method to extract histological features from H&E images by pretrained ResNet-50 model. You can use the parameter: --Hismodel ResNet50. It takes ~1 min



#### Learn cell representations by cell module

This function automatically learns cell-module representations by heterogeneous graph learning. It takes ~5 mins for DLPFC_151507.

```
python ./stKeep/Cell_model.py --inputPath ./test_data/DLPFC_151507/
```
In running, the useful parameters:

* lr: defines learning rate parameters for learning cell representations. The default value is 0.02.

* lam: defines the importance of the two types of representations (i.e., hierarchical and Semantic). The default value is 0.1. You can adjust it from 0.1 to 0.3 by 0.05;

To reproduce the result, you should use the default parameters. 

#### Output file

* Semantic_representations.txt: Semantic representations for cells.
  
* Hierarchical_representations.txt: Hierarchical representations for cells.


### Step 3. Run gene module

#### Calculate the input data for the gene module

This function automatically calculates input data for the gene module, including the relations between genes and cells, the links between genes and clusters (or cell states), and gene-positive pairs. It takes ~6 mins to generate the above-described files. 

Note: please check there are 'Protein_protein_interaction_network.txt' and 'Gene_regulatory_network.txt' in the 'utilities' folder. If you can't find it, please download it from link https://drive.google.com/drive/folders/1RTb_gHcpLhnbRMHrqn8tBtynesq5g5DI?usp=drive_link; and then put them into 'utilities' folder. 

```
python ./stKeep/Preprocess_Gene_module.py --inputPath ./test_data/DLPFC_151507/
```

#### Learn gene representations by gene module

This function automatically learns gene-module representations by heterogeneous graph learning. It takes ~1 min for DLPFC_151507.

```
python ./stKeep/Gene_model.py --inputPath ./test_data/DLPFC_151507/
```

In running, the useful parameters:

* lr: defines learning rate parameters for learning gene representations. The default value is 0.05.

* attn_drop: defines the dropout rate for the attention. The default value is 0.2.
  
#### Output file

* Gene_module_representation.txt: gene representations for identifying gene-modules.


### Step 4. Run cell-cell communication module

#### Calculate the input data for the CCC module

This function automatically calculates input data for the CCC module, including the denoised and normalized gene expression for ligands and receptors.

```
source("./stKeep/Processing.R")
Preprocess_CCC_model(basePath = "./test_data/DLPFC_151507/", LRP_data = "./utilities/Uninon_Ligand_receptors.RData")
```

This function loads 4,257 unique ligand-receptor pairs, selects expressed ligands and receptors for further analysis, utilizes knn_smoothing method to denoise gene expression data, and applies the denoised and normalized data for inferring CCC through the CCC model. It takes ~3 mins to generate the above-described files. 

#### Learn ligand-receptor interaction strengths through the CCC module

This function automatically learns LRP interaction strength by the CCC module. It takes ~20 min for DLPFC_151507.

```
python ./stKeep/CCC_model.py --inputPath ./test_data/DLPFC_151507/
```

In running, the useful parameters:

* lr_cci: defines learning rate parameters. The default value of the three parameters is 0.001;

* tau: defines denotes the temperature parameter. The default value is 0.05.

#### Output file

* CCC_module_LRP_strength.txt: inferred CCC interaction strength for cells.
  

## Further analysis

Some functions from the R file named Processing.R (in the stKeep folder) are based on the output files by cell module, gene module, and CCC module for further analysis.

* Cell_modules: clustering and visualization for cell-modules by Seurat package.

```
#Generate pdf file including clustering and visualization 
source("./stKeep/Processing.R")
basePath       = "./test_data/DLPFC_151507/"
MP_rep         = as.matrix(read.table( paste0(basePath, "stKeep/Semantic_representations.txt"), header = T, row.names = 1))
SC_rep         = as.matrix(read.table( paste0(basePath, "stKeep/Hierarchical_representations.txt"), header = T, row.names = 1))
Cell_obj       = Cell_modules(basePath, cbind(MP_rep, SC_rep), 7, basePath, "stKeep/stKeep_cell_clustering.pdf" )
```

* Gene_modules: identification of gene-gene relations from gene-modules; and Molecular_network: for visualization of gene-gene relations for each cluster.

```
#Generate cluster-specific gene-gene interactions.
gene_rep     = as.matrix(read.table( paste0(basePath, "stKeep/Gene_module_representation.txt"), header = T, row.names = 1))
Gene_obj     = Gene_modules(Cell_obj, gene_rep, 7, basePath, "stKeep/stKeep_gene_clustering.pdf"  )
TF_TG_links  = Molecular_network(Gene_obj, basePath, "stKeep/stKeep_molecular_network.pdf"  )
```

* CCC_modules: identification of CCC patterns from CCC-modules
  
```
#Generate cluster-specific CCC patterns.
## CCC patterns
featues      = c("RELN->ITGB1", "PENK->ADRA2A")
LR_activ     = as.matrix(read.table( paste0(basePath, "stKeep/CCC_module_LRP_strength.txt"), header = T, row.names = 1))
Cell_obj     = CCC_modules(Cell_obj, LR_activ, featues, basePath, "stKeep/CCC_patterns.pdf" )
```

* ......


# References

* GAT: https://github.com/gordicaleksa/pytorch-GAT

* SimCLR: https://github.com/google-research/simclr

* stLearn: https://github.com/BiomedicalMachineLearning/stLearn

* KNN_smoothing: https://github.com/yanailab/knn-smoothing
  

# Citation

Chunman Zuo* and Luonan Chen*. Dissecting tumor microenvironment from spatially resolved transcriptomics data by heterogeneous graph learning. 2023. (submitted).


