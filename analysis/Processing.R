library("Seurat")
library('ggplot2')
library('Matrix')
library('ggrepel')
library('igraph')

plot_colors=c("0" = "#6D1A9C", "1" = "#CC79A7","2"  = "#7495D3", "3" = "#59BE86", "4" = "#56B4E9", "5" = "#FEB915", 
              "6" = "#DB4C6C", "7" = "#C798EE", "8" = "#3A84E6", "9"= "#FF0099FF", "10" = "#CCFF00FF",
              "11" = "#268785", "12"= "#FF9900FF", "13"= "#33FF00FF", "14"= "#AF5F3C", "15"= "#DAB370", 
              "16" = "#554236", "17"= "#787878", "18"= "#877F6C")

suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(rsvd))

randomized_pca <- function(tmat, d, seed){
  set.seed(seed)
  rpca_obj <- rpca(tmat, k=d, center=T, scale=F, retx=T, p=10, q=7)
  rpca_obj$x
}

normalization_median <- function(mat){
  num_transcripts <- Matrix::colSums(mat)
  size_factor <- median(num_transcripts, na.rm = T) / num_transcripts
  t(t(mat) * size_factor)
}

freeman_tukey_transform <- function(mat){
  sqrt(mat) + sqrt(mat + 1)
}

pdist <- function(tmat){
  mtm <- Matrix::tcrossprod(tmat)
  sq <- rowSums(tmat^2)
  out0 <- outer(sq, sq, "+") - 2 * mtm
  out0[out0 < 0] <- 0
  sqrt(out0)
}

smoother_aggregate_nearest_nb <- function(mat, D, k){
  sapply(seq_len(ncol(mat)), function(cid){
    nb_cid <- head(order(D[cid, ]), k)
    closest_mat <- mat[, nb_cid, drop=FALSE]
    return(Matrix::rowSums(closest_mat))
  })
}

knn_smoothing <- function(mat, k, latent_matrix, seed=42){
  #' @param mat A numeric matrix with gene names on rows and cell names on columns.
  #' @param k Number of nearest neighbours to aggregate.
  #' @param latent_matrix low-representation matrix with sample by feature.
  #' @param seed Seed number. (default=42)
  #' @return A smoothed numeric matrix.
  cname     = colnames(mat)
  gname     = rownames(mat)
  num_steps = ceiling(log2(k + 1))
  S         = mat
  D         = pdist(latent_matrix)
  S         = smoother_aggregate_nearest_nb(mat, D, 15)
  colnames(S) = cname
  rownames(S) = gname
  return(S)
}

Preprocess_CCC_model <- function(basePath = "../test_data/DLPFC_151507/", utili_path = "../utilities/", LRP_data = "Uninon_Ligand_receptors.RData"){
  load(paste0(utili_path, LRP_data))
  idc             = Load10X_Spatial(data.dir= basePath )
  anno            = read.table(paste0(basePath,"151507_annotation.txt"), header = T, row.names = 1 )
  idc$Annotation  = anno$Layer[match(colnames(idc), row.names(anno))]
  match_int       = which(!is.na(idc$Annotation))
  spot_d          = colnames(idc)[match_int]
  sub_idc         = subset(idc, cells = spot_d)
  represen_data   = as.matrix(read.table(paste0(basePath,"stKeep/Semantic_representations.txt"), header = T, row.names = 1))
  represen_data1  = as.matrix(read.table(paste0(basePath,"stKeep/Hierarchical_representations.txt"), header = T, row.names = 1))
  spot_locs       = read.table(paste0(basePath,"stKeep/Spot_location.txt"),header = T, row.names = 1)
  latent_fea      = cbind(represen_data,represen_data1)[match(colnames(sub_idc), row.names(represen_data)),]
  mat             = as.matrix(sub_idc@assays$Spatial@counts)
  mat_smooth      = knn_smoothing( mat, 3, latent_fea )
  sub_idc@assays$Spatial@counts = mat_smooth
  sub_idc  = SCTransform(sub_idc, assay = "Spatial", verbose = FALSE)
  
  mat_it          = apply(mat, 1, function(x){length(which(x>0))})
  aa              = intersect(uni_ligand, row.names(mat)[which(mat_it>=5)])
  bb              = intersect(uni_receptor, row.names(mat)[which(mat_it>=5)])
  used_li         = used_re = NULL
  for(z in 1:length(uni_ligand))
  {
    if((is.element(uni_ligand[z], aa)) && (is.element(uni_receptor[z], bb)))
    {
      used_li = c(used_li, uni_ligand[z])
      used_re = c(used_re, uni_receptor[z])
    }
  }
  LR_pair = paste(used_li, "+", used_re, sep="")
  uniq_LR = unique(LR_pair)
  used_ligands = used_receptors = NULL
  for(m in 1:length(uniq_LR))
  {
    temps          = unlist(strsplit(uniq_LR[m], "[+]"))
    used_ligands   = c(used_ligands, temps[1])
    used_receptors = c(used_receptors, temps[2])
  }
  data           = as.matrix(sub_idc@assays$SCT@data)
  liagand_exps   = t(data[match(used_ligands, row.names(data)),])
  recep_exps     = t(data[match(used_receptors, row.names(data)),])
  
  liagand_exps_n = apply(liagand_exps, 2, function(x){(x-min(x))/(max(x)-min(x))})
  recep_exps_n   = apply(recep_exps, 2, function(x){(x-min(x))/(max(x)-min(x))})
  
  colnames(liagand_exps_n) = colnames(recep_exps_n) = unique(paste(used_li, "->", used_re, sep=""))
  write.table(liagand_exps_n[match(row.names(spot_locs), row.names(liagand_exps_n)),], file = paste0(basePath, "stKeep/ligands_expression.txt"), sep = "\t", quote = F)
  write.table(recep_exps_n[match(row.names(spot_locs), row.names(recep_exps_n)),], file = paste0(basePath, "stKeep/receptors_expression.txt"), sep = "\t", quote = F)
}

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
  p1  = DimPlot(Cell_obj,  label = T, label.size = 6, pt.size=1.5, cols = plot_colors)+
    theme(legend.position = "none",legend.title = element_blank())+ggtitle("")
  print(p1)
  p2  = SpatialDimPlot(Cell_obj, label = T, label.size = 3, cols = plot_colors)+
    theme(legend.position = "none",legend.title = element_blank())+ggtitle("")
  print(p2)
  dev.off()
  
  return(Cell_obj)
}

Gene_modules <- function(Cell_obj, Gene_rep, nCluster = 7, save_path = NULL, pdf_file = NULL ){
  inter_genes = c("SNAP25", "PVALB","ENC1", "HPCAL1", "MOBP", "MBP", "CARTPT", "FABP7", "AQP4", "KRT17")
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
  Gene_obj = RunUMAP(Gene_obj, reduction = "pca", dims = 1:dim(Gene_rep)[2], verbose = FALSE )
  umap_emd = as.matrix(Gene_obj@reductions$umap@cell.embeddings)
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


Molecular_network <- function(Gene_obj, utili_path = "../utilities/", save_path = NULL, pdf_file = NULL ){
  uniqu_cl    = unique(as.character(Idents(Gene_obj)))
  gr_network  = readRDS(paste0(utili_path, "gr_network.rds"))
  from_TG     = to_TG  =  list()
  pdf(paste0( save_path, pdf_file ), width = 20, height = 20)
  for(z in 1:length(uniqu_cl))
  {
    temp_gens = colnames(Gene_obj)[which(Idents(Gene_obj)==uniqu_cl[z])]
    temp_tfs  = intersect(temp_gens, gr_network[[1]])
    temp_tgs  = intersect(temp_gens, gr_network[[2]])
    from_list = to_list = NULL
    for(zz in 1:length(temp_tfs))
    {
      tes       = intersect(temp_tgs, gr_network[[2]][which(gr_network[[1]]==temp_tfs[zz])])
      from_list = c(from_list, rep(temp_tfs[zz], length(tes)))
      to_list   = c(to_list, tes)
    }
    if((length(from_list)>0)&&(length(to_list)>0))
    {
      unique_genes = unique(c(from_list,to_list))
      tf_types     = rep(0, length(unique_genes))
      tf_types[match(intersect(gr_network[[1]],unique_genes), unique_genes)] = 1
      actors    = data.frame(name=unique_genes, TF  = tf_types)
      relations = data.frame(from=from_list,  to=to_list)
      g         = graph_from_data_frame(relations, directed=TRUE, vertices=actors)
      coords <- layout_(g, on_sphere())
      
      V(g)[V(g)$TF == 1]$label.color = "red"
      V(g)[V(g)$TF == 0]$label.color = "blue"
      
      plot(g, layout = coords,  vertex.label = unique_genes, vertex.shape="none",
           vertex.label.font=9, vertex.label.cex=0.8, main = paste(uniqu_cl[z], "-", "GRN", sep=""))
    }
    from_TG[[z]] = from_list
    to_TG[[z]]   = to_list
  }
  dev.off()
  
  names(from_TG) = uniqu_cl
  names(to_TG)   = uniqu_cl
  return( list(from_TG, to_TG) )
}


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

CCC_modules <- function(Cell_obj, LRP_activ, featues, save_path = NULL, pdf_file = NULL ){
  colnames(LRP_activ) = reNames(colnames(LRP_activ))
  LRP_activi   = t(LRP_activ[match(colnames(Cell_obj), row.names(LRP_activ)),])
  LRP_activity = apply(LRP_activi, 1, function(x){(x-min(x, na.rm = T))/(max(x,na.rm = T)-min(x,na.rm = T))})
  LR_assay = CreateAssayObject(counts = t(LRP_activity))
  Cell_obj[["LRP_activity"]] = LR_assay
  DefaultAssay(Cell_obj)     = "LRP_activity"
  #Diff_LRPs        = FindAllMarkers(Cell_obj, only.pos = T )
  pdf(paste0( save_path, pdf_file ), width = 10, height = 10)
  p1 = SpatialFeaturePlot(Cell_obj, features = featues, ncol = 2)
  plot(p1)
  dev.off()
  return(Cell_obj)
}
