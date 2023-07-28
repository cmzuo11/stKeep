# Dissecting tumor microenvironment from spatially resolved transcriptomics data by heterogeneous graph learning.

![image](https://github.com/cmzuo11/stKeep/blob/main/utilities/main-framework.png)

Overview of the stKeep model. a-c Given each SRT data with four-layer profiles: histological images (I), spatial locations (S), gene expression (X), histological regions (Y), and gene-gene interactions such as PPI, GRN, and LRP as the input, stKeep integrates them to construct HG for dissecting tumor ecosystems. d Cell module adopts a cell-centered HG to capture local hierarchical representations (〖R_i〗^1) through aggregating features from genes and regions by attention, while leveraging intercellular graphs including SLG, HSG, and TSG to learn global semantic representations (〖R_i〗^2), and collaboratively integrates two representations by self-supervised learning. e Gene module utilizes a gene-centered HG to learn low-dimensional representations by combining features from cells and clusters using attention, while ensuring co-relational gene pairs are embedded adjacent to each other using contrastive learning. f Cell-cell communication module leverages attention-based heterogeneous graphs to infer ligand-receptor interaction strength (H_i) for each cell by aggregating ligand information from the neighbors for a central cell, while guaranteeing that CCC patterns can characterize diverse cell-states within TME. Note that each graph indicates one LRP. g The unified framework with three modules (d-f) can be used to dissect tumor ecosystems by detecting spatial clusters and visualizing them, identifying cell-state-specific gene-modules and receptor-gene interaction networks, and inferring cellular communication strength.

# Installation

## Install stKeep

Installation was tested on Red Hat 7.6 with Python 3.6.12 and torch 1.6.0 on a machine with one 40-core Intel(R) Xeon(R) Gold 5115 CPU addressing with 132GB RAM, and two NVIDIA TITAN V GPU addressing 24GB. stKeep is implemented in the Pytorch framework. Please run stKeep on CUDA if possible. 

#### 1. Grab the source code of stKeep

```
git clone https://github.com/cmzuo11/stKeep.git

cd stKeep
```

#### 2. Install stKeep in the virtual environment by conda 

* Firstly, install conda: https://docs.anaconda.com/anaconda/install/index.html

* Then, automatically install all used packages (described by "used_package.txt") for stKeep in a few mins.

```
conda create -n stKeep python=3.6.12 pip

source activate

conda activate stKeep

pip install -r used_package.txt
```

## Install R packages 

* Install tested on R =4.0.0

* Install package 'Seurat' based on the Github https://github.com/satijalab/seurat

* install.packages("ggplot2")

# Quick start

## Input

* A general output of 10X pipeline, for example, a directory includes a file named filtered_feature_bc_matrix.h5, a directory named spatial with five files: tissue_positions_list.csv, tissue_lowres_image.png, tissue_hires_image.png, metrics_summary_csv.csv, scalefactors_json.json, and a directory named as filtered_feature_bc_matrix with three files: matrix.mtx.gz, features.tsv.gz, and barcodes.tsv.gz;  

* Take slice 151507 as an example, you can download it by the following scripts:

```
wget https://zenodo.org/record/7244758/files/stKeep_test_data.zip

unzip stKeep_test_data.zip
```

Note: The folder named 'DLPFC_151507' contains the raw data of slice 151507.

## Run

### Step 1. Preprocess raw data

This function automatically (1) learns 50-dimensional features from 2000 highly variable genes of gene expression data, (2) trains SimCLR model (500 iterations) by data augmentations and contrastive learning and extracts 2048-dimensional visual features from histological data, and (3) saves the physical location of each spot into a file 'Spot_location.csv' into a folder named spatial of the current directory.

```
python Preprcessing_stKeep.py --basePath ./stKeep_test_data/DLPFC_151507/ 
```

The running time mainly depends on the iteration of SimCLR training. It takes 3.7h to generate the above-described files. You can modify the following parameters to reduce time:

* batch_size_I: defines the batch size for training SimCLR model. The default value is 128. You can modify it based on your memory size. The larger the parameter, the less time.

* max_epoch_I: defines the max iteration for training SimCLR model. The default value is 500. You can modify it. The smaller the parameter, the less time.

To reproduce the result, you should use the default parameters.

Note: To reduce your waiting time, we have uploaded our preprocessed data into the folder ./stKeep_test_data/DLPFC_151507/stKeep/. You can directly perform step 3.

### Step 2. Manual region segmentation (for IDC dataset)

You can adapt two different methods to define the class of each spot:

#### method 1 

* install the Loupe browser software from the website: https://www.10xgenomics.com/products/loupe-browser/downloads, and apply it to manually annotate cells based on our previously defined region segmentation.

#### method 2
  
* install the labelme software from the link on Github: https://github.com/wkentaro/labelme, and apply it to manually outline each tumor region based on our defined strategy, and save the annotation into a json file named 'tissue_hires_image.json' of a directory named image_segmentation.

* Define the classification for each spot based on above-generated json file. Here, we use IDC dataset as an example.

```
python Image_cell_segmentation.py --basePath ./stKeep_test_data/IDC/ --jsonFile tissue_hires_image.json
```
Note: To reduce your waiting time, we have uploaded the tissue_hires_image.json and the processed result from step 1 into a folder named IDC. You can directly perform step 3.

### Step 3. Run stKeep model

This function automatically learns cell-module representations by heterogeneous graph learning. It takes ~7 mins for DLPFC_151507 and ~9 mins for IDC.

```
python stKeep_model.py --basePath ./stKeep_test_data/DLPFC_151507/
```
In running, the useful parameters:

* lr_T1 for HSG, lr_T2 for SLG, lr_T3 for collaborative learning: defines learning rate parameters for learning view-specific representations by single-view graph and robust representations by multi-view graph. i.e., . The default value of the three parameters is 0.002. You can adjust them from 0.001 to 0.003 by 0.001;

* max_epoch_T: defines the max iteration for training view-specific graph or multi-view graphs. The default value is 500. You can modify it. The larger the parameter, the more time.

* beta_pa: defines the penalty for the knowledge transfer from robust representations to view-specific representations. The default value is 8.

* knn: defines the K-nearest similarity spots for each spot to construct HSG or SLG. The default value is 7 where the K-nearest spots for a spot include itself.

* latent_T1 and latent_T2 define the dimension of two layers of GAT for SGATE model. Here, the default value of the DLPFC and IDC datasets is 25 and 10, 32 and 16, respectively.

* fusion_type: definies the multi-view graph fusion types.

To reproduce the result, you should use the default parameters.

## Output

## Output file will be saved for further analysis:

* GAT_2-view_model.pth: a saved model for reproducing results.

* GAT_2-view_robust_representation.csv: robust representations for latter clustering, visualization, and data denoising.

## Further analysis

Some functions from R file named Postprocessing.R (in the stKeep folder) are based on the file named GAT_2-view_robust_representation.csv for further analysis.

* Seurat_processing: clustering, visualization, and differential analysis by Seurat package.

```
#Generate pdf file includes clustering and visualization 
library('Seurat')
library('ggplot2')
source(./stKeep/Postprocessing.R)
basePath       = "./stKeep_test_data/DLPFC_151507/"
robust_rep     = read.csv( paste0(basePath, "stKeep/Cell_model_representation.csv"), header = T, row.names = 1)
Seurat_obj     = Seurat_processing(basePath, robust_rep, 10, 7, basePath, "stKeep/stKeep_cell_clustering.pdf" )
```

* knn_smoothing: data denoising by its three nearest neighboring spots that are calculated based on the distance of robust representations between any two spots.

```
#data denoising based on three nearest neighboring spots
input_features = as.matrix(robust_rep[match(colnames(Seurat_obj), row.names(robust_rep)),])
Seurat_obj     = FindVariableFeatures(Seurat_obj, nfeatures=2000)
hvg            = VariableFeatures(Seurat_obj)
rna_data       = as.matrix(Seurat_obj@assays$Spatial@counts)
hvg_data       = rna_data[match(hvg, row.names(rna_data)), ]

mat_smooth     = knn_smoothing( hvg_data, 3, input_features )
colnames(mat_smooth) = colnames(Seurat_obj)

#find spatially variable genes
Seurat_smooth         = CreateSeuratObject(counts=mat_smooth, assay='Spatial')
Idents(Seurat_smooth) = Idents(Seurat_obj)

Seurat_smooth = SCTransform(Seurat_smooth, assay = "Spatial", verbose = FALSE)
top_markers   = FindAllMarkers(Seurat_smooth, assay='SCT', slot='data', only.pos=TRUE) 
```

* ......

# References

* GAT: https://github.com/gordicaleksa/pytorch-GAT

* SimCLR: https://github.com/google-research/simclr

* stLearn: https://github.com/BiomedicalMachineLearning/stLearn

* KNN_smoothing: https://github.com/yanailab/knn-smoothing

# Citation

Chunman Zuo* and Luonan Chen*. Dissecting tumor microenvironment from spatially resolved transcriptomics data by heterogeneous graph learning. 2023. (submitted).


