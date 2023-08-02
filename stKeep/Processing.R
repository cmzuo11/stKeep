library("Seurat")
library('ggplot2')
library('Matrix')
library('ggrepel')
library('igraph')
library("pheatmap")

plot_colors=c("0" = "#6D1A9C", "1" = "#CC79A7","2"  = "#7495D3", "3" = "#59BE86", "4" = "#56B4E9", "5" = "#FEB915", 
              "6" = "#DB4C6C", "7" = "#C798EE", "8" = "#3A84E6", "9"= "#FF0099FF", "10" = "#CCFF00FF",
              "11" = "#268785", "12"= "#FF9900FF", "13"= "#33FF00FF", "14"= "#AF5F3C", "15"= "#DAB370", 
              "16" = "#554236", "17"= "#787878", "18"= "#877F6C")

Cell_modules <- function(basePath, robust_rep, nCluster = 7, save_path = NULL, pdf_file = NULL ){
  idc = Load10X_Spatial(data.dir= basePath )
  idc = SCTransform(idc, assay = "Spatial", verbose = FALSE)
  idc = RunPCA(idc, assay = "SCT", verbose = FALSE, npcs = 100)
  
  inter_c = intersect(row.names(robust_rep), colnames(idc))
  Cell_obj = subset(idc, cells =inter_c)      
  in_feas = robust_rep[match(colnames(Cell_obj), row.names(robust_rep)),]
  Cell_obj@reductions$pca@cell.embeddings[,1:dim(in_feas)[2]] = in_feas
  Cell_obj = FindNeighbors(Cell_obj, reduction = "pca", dims = 1:dim(in_feas)[2])
  
  for(qq in seq(0.05,1.5,0.01))
  {
    Cell_obj = FindClusters( Cell_obj, resolution = qq,  verbose = FALSE )
    if(length(table(Idents(Cell_obj)))==nCluster)
    {
      break
    }
  }
  
  Cell_obj = RunUMAP(Cell_obj, reduction = "pca", dims = 1:dim(in_feas)[2])

  pdf( paste0( save_path, pdf_file ), width = 10, height = 10)
  p1  = DimPlot(Cell_obj, reduction = "umap", label = T, label.size = 6, pt.size=1.5, cols = plot_colors)+
        theme(legend.position = "none",legend.title = element_blank())+ggtitle("")
  print(p1)
  p2  = SpatialDimPlot(Cell_obj, label = T, label.size = 3, cols = plot_colors)+
        theme(legend.position = "none",legend.title = element_blank())+ggtitle("")
  print(p2)
  dev.off()
  
  return(Cell_obj)
}

Gene_modules <- function(Cell_obj, Gene_rep, nCluster = 7, save_path = NULL, pdf_file = NULL ){
  inter_genes = c("SNAP25", "MOBP", "PCP4", "FABP7", "PVALB", "CCK", "ENC1", "AQP4", "TRABD2A", "HPCAL1", "FREM3", "KRT17")
  count       = as.matrix(Cell_obj@assays$Spatial@counts)
  count_new   = t(count[match(row.names(Gene_rep), row.names(count) ),])
  Gene_obj    = CreateSeuratObject(counts = count_new)
  Gene_obj    = FindVariableFeatures(Gene_obj, selection.method = "vst", nfeatures = dim(count_new)[1])
  Gene_obj    = ScaleData(Gene_obj, verbose = FALSE)
  Gene_obj    = RunPCA(Gene_obj, npcs = 50, verbose = FALSE)
  datss       = as.matrix(Gene_obj@assays$RNA@data)
  Gene_obj@reductions$pca@cell.embeddings = Gene_rep
  Gene_obj = FindNeighbors(Gene_obj, reduction = "pca", verbose = FALSE, dims = 1:dim(Gene_rep)[2])
  
  for(z in seq(0.01,2,0.01))
  {
    Gene_obj <- FindClusters( Gene_obj, resolution = z,  verbose = FALSE )
    if(length(table(Idents(Gene_obj)))==nCluster)
    {
      break
    }
  }
  Gene_obj = RunUMAP(Gene_obj, reduction = "pca", dims = 1:dim(Gene_rep)[2], verbose = FALSE, reduction.name = "umap_rna" )

  umap_emd = as.matrix(Gene_obj@reductions$umap_rna@cell.embeddings)
  labe_not = rep(FALSE, dim(Gene_obj)[2])
  labe_not[match(inter_genes, colnames(Gene_obj))] = T
  color_not = rep(FALSE, dim(Gene_obj)[2])
  color_not[match(inter_genes, colnames(Gene_obj))] = T
  df_info  = data.frame(Genes = colnames(Gene_obj), umap_1 = umap_emd[,1], umap_2 = umap_emd[,2], Label = labe_not, color = color_not)
  row.names(df_info) = colnames(Gene_obj)
  
  pdf( paste0( save_path, pdf_file ), width = 10, height = 10)
  p1 = DimPlot(Gene_obj, reduction = "umap", label = T, label.size = 6, pt.size=1.5, cols = plot_colors)+
       theme(legend.position = "none",legend.title = element_blank())+ggtitle("")
  plot(p1)
  p2 = ggplot(df_info) +
       geom_point(aes(x = umap_1, y = umap_2, col = color)) +
       geom_text_repel(aes(x = umap_1, y = umap_2, label = ifelse(Label == T, rownames(df_info),"")),
                       max.overlaps = Inf)
  plot(p2)
  dev.off()
  
  return(Gene_obj)
}

Molecular_network <- function(Gene_obj, save_path = NULL, pdf_file = NULL ){
  uniqu_cl = unique(as.character(Idents(Gene_obj)))
  load("gr_network")
  load("ppi_matrix")
  load("LRP")
  pdf(paste0( save_path, pdf_file ), width = 15, height = 15)
  
  for(z in 1:length(uniqu_cl))
  {
    temp_gens = colnames(Gene_obj)[which(Idents(Gene_obj)==uniqu_cl[z])]
    temp_tfs  = intersect(temp_gens, gr_network[[1]])
    temp_tgs  = intersect(temp_gens, gr_network[[2]])
    
    from_list = to_list= NULL
    for(zz in 1:length(temp_tfs))
    {
      tes       = intersect(temp_tgs, gr_network[[2]][which(gr_network[[1]]==temp_tfs[zz])])
      from_list = c(from_list, rep(temp_tfs[zz], length(tes)))
      to_list   = c(to_list, tes)
    }
    
    temp_ppis = intersect(ppi_matrix[[1]], intersect(temp_gens, uni_receptor))
    from_ppi  = to_ppi = NULL
    for(zz in 1:length(temp_ppis))
    {
      tes       = intersect(temp_gens, ppi_matrix[[2]][which(ppi_matrix[[1]]==temp_ppis[zz])])
      from_ppi  = c(from_ppi, rep(temp_ppis[zz], length(tes)))
      to_ppi    = c(to_ppi, tes)
    }
    
    if((length(from_list)>0)&&(length(to_list)>0))
    {
      unique_genes = unique(c(from_list,to_list))
      tf_types     = rep(0, length(unique_genes))
      tf_types[match(intersect(gr_matrix[[1]],unique_genes), unique_genes)] = 1
      
      actors    = data.frame(name=unique_genes, TF  = tf_types)
      relations = data.frame(from=from_list,  to=to_list)
      g         = graph_from_data_frame(relations, directed=TRUE, vertices=actors)
      coords    = layout_(g, in_circle())
      
      V(g)[V(g)$TF == 1]$label.color = "red"
      V(g)[V(g)$TF == 0]$label.color = "blue"
      
      plot(g, layout = coords,  vertex.label = unique_genes, vertex.shape="none",
           vertex.label.font=9, vertex.label.cex=0.8, main = paste(uniqu_cl[z], "-", "GRN", sep=""))
    }
    
    if((length(from_ppi)>0)&&(length(to_ppi)>0))
    {
      unique_genes = unique(c(from_ppi,to_ppi))
      tf_types     = rep(0, length(unique_genes))
      tf_types[match(intersect(ppi_matrix[[1]], intersect(unique_genes, uni_receptor)), unique_genes)] = 1
      
      actors    = data.frame(name=unique_genes, TF  = tf_types)
      relations = data.frame(from=from_ppi,  to=to_ppi)
      g         = graph_from_data_frame(relations, directed=TRUE, vertices=actors)
      coords    = layout_(g, in_circle())
      
      V(g)[V(g)$TF == 1]$label.color = "red"
      V(g)[V(g)$TF == 0]$label.color = "blue"
      
      plot(g, layout = coords,  vertex.label = unique_genes, vertex.shape="none",
           vertex.label.font=9, vertex.label.cex=0.8, main = paste(uniqu_cl[z], "-", "LR", sep=""))
    }
  }
  dev.off()
}


CCC_modules <- function(Cell_obj, LRP_activ, featues, save_path = NULL, pdf_file = NULL ){
  
  LRP_activi   = t(LRP_activ[match(colnames(Cell_obj), row.names(LRP_activ)),])
  LRP_activity = apply(LRP_activi, 1, function(x){(x-min(x, na.rm = T))/(max(x,na.rm = T)-min(x,na.rm = T))})
  
  LR_assay = CreateAssayObject(counts = t(LRP_activity))
  Cell_obj[["LRP_activity"]] = LR_assay
  DefaultAssay(Cell_obj)     = "LRP_activity"
  
  Idents(Cell_obj) = Cell_obj$Annotation
  Diff_LRPs        = FindAllMarkers(Cell_obj, only.pos = T )
  
  pdf(paste0( save_path, pdf_file ), width = 10, height = 10)
  p1 = SpatialFeaturePlot(Cell_obj, features = featues, ncol = 2)
  plot(p1)
  dev.off()
  
  return(Cell_obj)
}


currPath  = "/sibcb2/chenluonanlab7/cmzuo/workPath/CMSSL/spatial_result/DLPFC/151507/stKeep/"
load("/sibcb2/chenluonanlab7/cmzuo/workPath/CMSSL/spatial_result/DLPFC/151507/151507_100.RData")
match_int = which(!is.na(idc$Annotation))
spot_d    = colnames(idc)[match_int]
sub_idc   = subset(idc, cells = spot_d)

represen_data   = as.matrix(read.table(paste0(currPath,"Semantic_representations.txt"), header = T, row.names = 1))
input_features  = represen_data[match(colnames(sub_idc), row.names(represen_data)),]
represen_data1  = as.matrix(read.table(paste0(currPath, "Hierarchical_representations.txt"), header = T, row.names = 1))
input_features1 = represen_data1[match(colnames(sub_idc), row.names(represen_data1)),]
combine_featus  = cbind(input_features,input_features1)

original_pca_emb                   = sub_idc@reductions$pca@cell.embeddings
row.names(combine_featus)          = row.names(original_pca_emb)
sub_idc@reductions$pca@cell.embeddings[,1:dim(combine_featus)[2]] = combine_featus
sub_idc = FindNeighbors(sub_idc, reduction = "pca", dims = 1:dim(combine_featus)[2], verbose = FALSE)

sub_idc = RunUMAP(sub_idc, reduction = "pca", dims = 1:dim(combine_featus)[2], verbose = FALSE )
sub_idc = FindClusters( sub_idc, resolution = 0.15 )

plot_colors_B=c("5" = "#DB4C6C", "3" = "#FEB915", "2" = "#56B4E9", "4" = "#59BE86", "0" = "#7495D3",
                "6" = "#CC79A7","1" ="#6D1A9C")

pdf(paste(currPath, "HIN_clustering.pdf",sep=""), width = 10, height = 10)
p2  = SpatialDimPlot(sub_idc, label = F, label.size = 3, cols = plot_colors_B )
plot(p2)
p2  = DimPlot(sub_idc, label = F, label.size = 3, cols = plot_colors_B)
plot(p2)
dev.off()


mp_file = c("Semantic_representations.txt")
sc_file = c("Hierarchical_representations.txt")

temp_file = c(list.files(path = currPath, pattern = "Semantic_representations-"))

for(zzz in 1:length(temp_file))
{
  temp_pattern    = unlist(strsplit(temp_file[zzz], "Semantic_representations"))

  represen_data   = as.matrix(read.table(paste0(currPath,temp_file[zzz]), header = T, row.names = 1))
  input_features  = represen_data[match(colnames(sub_idc), row.names(represen_data)),]
  
  represen_data1  = as.matrix(read.table(paste(currPath, "Hierarchical_representations", temp_pattern[2],sep=""), header = T, row.names = 1))
  input_features1 = represen_data1[match(colnames(sub_idc), row.names(represen_data1)),]
  
  pdf(paste(currPath, "HIN_clustering_mp_sc-check", temp_pattern[2], ".pdf",sep=""), width = 10, height = 10)
  plot_clustering_data(sub_idc, input_features)
  plot_clustering_data(sub_idc, input_features1)
  plot_clustering_data(sub_idc, cbind(input_features,input_features1))
  dev.off()
}


represen_data   = as.matrix(read.table(paste0(currPath,"Cell_encoding_AE.txt"), header = T, row.names = 1))
input_features  = represen_data[match(colnames(sub_idc), row.names(represen_data)),]
pdf(paste(currPath, "AE_cl.pdf",sep=""), width = 10, height = 10)
plot_clustering_data(sub_idc, input_features)
dev.off()

represen_data   = as.matrix(read.table("/sibcb2/chenluonanlab7/cmzuo/workPath/Spatial_check/DLPFC/151507/HVGs_2000_AE_50.txt"))
pdf(paste(currPath, "AE_cl.pdf",sep=""), width = 10, height = 10)
plot_clustering_data(sub_idc, represen_data)
dev.off()


## gene modules
cancer_cl   = c("WM", "Layer6", "Layer5", "Layer4", "Layer3", "Layer2", "Layer1")
bk = c(seq(0, 0.4,by=0.001),seq(0.401, 1,by=0.001))
library('igraph')
library("pheatmap")

gene_latent = as.matrix(read.table(paste0(currPath, "Gene_module_representation.txt"), header = T, row.names = 1))
count_new   = t(as.matrix(sub_idc@assays$Spatial@counts))[,match(row.names(gene_latent), row.names(sub_idc) )]

sub_pbmc    = CreateSeuratObject(counts = count_new)
sub_pbmc    = FindVariableFeatures(sub_pbmc, selection.method = "vst", nfeatures = dim(count_new)[1])

sub_pbmc = SCTransform(sub_pbmc, verbose = T)
sub_pbmc = NormalizeData(sub_pbmc,  verbose = FALSE)

sub_pbmc    = ScaleData(sub_pbmc, verbose = FALSE)
sub_pbmc    = RunPCA(sub_pbmc, npcs = 50, verbose = FALSE)
datss       = as.matrix(sub_pbmc@assays$RNA@data)

original_pca_emb        = sub_pbmc@reductions$pca@cell.embeddings
row.names(gene_latent)  = row.names(original_pca_emb)
sub_pbmc@reductions$pca@cell.embeddings[,1:dim(gene_latent)[2]] = gene_latent
sub_pbmc = FindNeighbors(sub_pbmc, reduction = "pca", dims = 1:dim(gene_latent)[2], verbose = FALSE)

sub_pbmc     = FindNeighbors(sub_pbmc, reduction = "pca", verbose = FALSE, dims = 1:50)
sub_pbmc     = FindClusters( sub_pbmc, resolution = 0.6,  verbose = FALSE )
sub_pbmc     = RunUMAP(sub_pbmc, reduction = "pca", dims = 1:50, verbose = FALSE, reduction.name = "umap_rna" )
uniqu_cl     = unique(as.character(Idents(sub_pbmc)))
ac_mean      =  Gene_mean_2(t(datss), as.character(Idents(sub_pbmc)), as.character(sub_idc$Annotation), cancer_cl)
pdf(paste(currPath, "HIN_model_clustering_new.pdf",sep=""), width = 10, height = 10)
pheatmap(ac_mean, cluster_rows = F, cluster_cols = F, scale="row",main="", cex.main=2, breaks=bk,
         show_rownames=T, treeheight_row = F, angle_col= 90, fontsize_row = 15, fontsize_number = 15, fontsize_col = 15,
         color=c(colorRampPalette(colors = c("blue","white"))(length(bk)/2),colorRampPalette(colors = c("white","red"))(length(bk)/2)) )
p1 = DimPlot(sub_pbmc, reduction = "umap_rna", label = T, label.size = 4)+
  ggtitle("")+ theme_classic(base_size = 15)+ theme(legend.position = "none")
plot(p1)
dev.off()


temp_file = c(list.files(path = currPath, pattern = "Gene_module_representation-"))
count     = as.matrix(sub_idc@assays$Spatial@counts)
count_new = t(count[match(row.names(gene_latent), row.names(count) ),])
sub_pbmc    = CreateSeuratObject(counts = count_new)
sub_pbmc    = FindVariableFeatures(sub_pbmc, selection.method = "vst", nfeatures = dim(count_new)[1])
sub_pbmc    = ScaleData(sub_pbmc, verbose = FALSE)
sub_pbmc    = RunPCA(sub_pbmc, npcs = 50, verbose = FALSE)
datss       = as.matrix(sub_pbmc@assays$RNA@data)


for(z in 1:length(temp_file))
{
  temp_pattern    = unlist(strsplit(temp_file[z], "Gene_module_representation"))
  gene_latent = as.matrix(read.table(paste0(currPath, temp_file[z]), header = T, row.names = 1))
  sub_pbmc@reductions$pca@cell.embeddings[,1:dim(gene_latent)[2]] = gene_latent
  sub_pbmc = FindNeighbors(sub_pbmc, reduction = "pca", dims = 1:dim(gene_latent)[2], verbose = FALSE)
  sub_pbmc = FindNeighbors(sub_pbmc, reduction = "pca", verbose = FALSE, dims = 1:50)
  for(z in seq(0.01,2,0.01))
  {
    sub_pbmc <- FindClusters( sub_pbmc, resolution = z,  verbose = FALSE )
    if(length(table(Idents(sub_pbmc)))==7)
    {
      break
    }
  }
  sub_pbmc     = RunUMAP(sub_pbmc, reduction = "pca", dims = 1:50, verbose = FALSE, reduction.name = "umap_rna" )
  ac_mean      = Gene_mean_2(t(datss), as.character(Idents(sub_pbmc)), as.character(sub_idc$Annotation), cancer_cl)
  umap_emd = as.matrix(sub_pbmc@reductions$umap_rna@cell.embeddings)
  labe_not = rep(FALSE, dim(sub_pbmc)[2])
  labe_not[match(inter_genes, colnames(sub_pbmc))] = T
  color_not = rep(FALSE, dim(sub_pbmc)[2])
  color_not[match(inter_genes, colnames(sub_pbmc))] = T
  df_info  = data.frame(Genes = colnames(sub_pbmc), umap_1 = umap_emd[,1], umap_2 = umap_emd[,2], Label = labe_not, color = color_not)
  row.names(df_info) = colnames(sub_pbmc)
  pdf(paste(currPath, "HIN_gene_clustering-", temp_pattern[2],".pdf",sep=""), width = 10, height = 10)
  pheatmap(ac_mean, cluster_rows = F, cluster_cols = F, scale="row",main="", cex.main=2, breaks=bk,
           show_rownames=T, treeheight_row = F, angle_col= 90, fontsize_row = 15, fontsize_number = 15, fontsize_col = 15,
           color=c(colorRampPalette(colors = c("blue","white"))(length(bk)/2),colorRampPalette(colors = c("white","red"))(length(bk)/2)) )
  p1 = DimPlot(sub_pbmc, reduction = "umap_rna", label = T, label.size = 4)+
    ggtitle("")+ theme_classic(base_size = 15)+ theme(legend.position = "none")
  plot(p1)
  pp = ggplot(df_info) +
    geom_point(aes(x = umap_1, y = umap_2, col = color)) +
    geom_text_repel(aes(x = umap_1, y = umap_2, label = ifelse(Label == T, rownames(df_info),"")),
                    max.overlaps = Inf)
  plot(pp)
  dev.off()
}


inter_genes = c("SNAP25", "MOBP", "PCP4", "FABP7", "PVALB", "CCK", "ENC1", "AQP4", "TRABD2A", "HPCAL1", "FREM3", "KRT17")
inter_genes = unique(inter_genes)
temp_file = c(list.files(path = currPath, pattern = "Gene_module_representation-"))

for(zzz in 1:length(temp_file))
{
  input_features  = as.matrix(read.table(paste0(currPath,temp_file[zzz]), header = T, row.names = 1))
  original_pca_emb                   = sub_pbmc@reductions$pca@cell.embeddings
  sub_pbmc@reductions$pca@cell.embeddings = input_features
  
  sub_pbmc = FindNeighbors(sub_pbmc, reduction = "pca", verbose = FALSE, dims = 1:50)
  sub_pbmc = FindClusters( sub_pbmc, resolution = 0.6,  verbose = FALSE )
  sub_pbmc = RunUMAP(sub_pbmc, reduction = "pca", dims = 1:50, verbose = FALSE, reduction.name = "umap_rna" )
  
  temp_pattern    = unlist(strsplit(temp_file[zzz], "Gene_module_representation"))
  
  umap_emd = as.matrix(sub_pbmc@reductions$umap_rna@cell.embeddings)
  labe_not = rep(FALSE, dim(sub_pbmc)[2])
  labe_not[match(inter_genes, colnames(sub_pbmc))] = T
  color_not = rep(FALSE, dim(sub_pbmc)[2])
  color_not[match(inter_genes, colnames(sub_pbmc))] = T
  df_info  = data.frame(Genes = colnames(sub_pbmc), umap_1 = umap_emd[,1], umap_2 = umap_emd[,2], Label = labe_not, color = color_not)
  row.names(df_info) = colnames(sub_pbmc)
  
  pdf(paste(currPath, "HIN_sss_clustering_", temp_pattern[2], ".pdf",sep=""), width = 10, height = 10)
  p1 = DimPlot(sub_pbmc, reduction = "umap_rna", label = T)
  plot(p1)
  pp = ggplot(df_info) +
    geom_point(aes(x = umap_1, y = umap_2, col = color)) +
    geom_text_repel(aes(x = umap_1, y = umap_2, label = ifelse(Label == T, rownames(df_info),"")),
                    max.overlaps = Inf)
  plot(pp)
  dev.off()
}



###CCC
reNames <- function(Names)
{
  re_names = NULL
  for(z in 1:length(Names))
  {
    temps    = unlist(strsplit(Names[z], "[..]"))
    re_names = c(re_names, paste(temps[1], "->", temps[3],sep = ""))
  }
  return(re_names)
}

currPath  = "/sibcb2/chenluonanlab7/cmzuo/workPath/CMSSL/spatial_result/DLPFC/151507/stKeep/"
temp_file = c(list.files(path = currPath, pattern = "CCC_module_LRP_strength-"))
featues   = c("RELN->ITGB1", "PENK->ADRA2A")

for(z in 14:16)
{
  LR_activ     = as.matrix(read.table(paste0(currPath, temp_file[z]), header = T, row.names = 1))
  colnames(LR_activ) = reNames(colnames(LR_activ))
  temps = unlist(strsplit(temp_file[z], "CCC_module_LRP_strength"))
  
  LRP_activi   = t(LR_activ[match(colnames(sub_idc), row.names(LR_activ)),])
  LRP_activity = apply(LRP_activi, 1, function(x){(x-min(x, na.rm = T))/(max(x,na.rm = T)-min(x,na.rm = T))})
  
  LR_assay = CreateAssayObject(counts = t(LRP_activity))
  sub_idc[["LRP_activity"]] = LR_assay
  DefaultAssay(sub_idc)     = "LRP_activity"
  
  pdf(paste0( currPath, paste("CCC",temps[2],".pdf",sep="") ), width = 10, height = 10)
  p1 = SpatialFeaturePlot(sub_idc, features = featues, ncol = 2)
  plot(p1)
  dev.off()
}



LR_assay = CreateAssayObject(counts = t(LR_activ_1_n))
sub_idc[["LR_activity"]] = LR_assay
DefaultAssay(sub_idc)    = "LR_activity"

pdf(paste(currPath, "Inter_CCC_pattern.pdf", sep=""), width = 10, height = 10)
p1 = SpatialFeaturePlot(sub_idc, features = featues, ncol = 2)
plot(p1)
dev.off()


