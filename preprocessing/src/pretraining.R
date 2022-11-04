source('src/function.R')

path <- "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/TRAIN"
now_date <- Sys.time() %>% str_split(" ") %>% unlist() %>% .[1]
save_path <- paste0(path, "/", now_date)

dir.create(save_path, showWarnings = FALSE, recursive = TRUE)

# TCGA PANCAN for pretrain ####
tcga_preprocessing(save_path = save_path)

# DeepDEP input source CCLE ----
ccle_preprocessing(save_path = save_path, DepOI_SD = 0)

# Final, CCLE-TCGA intersection ----
# TCGA index RData load
load(paste0(save_path, "/TCGA-PANCAN-EXPRESSION_index.RData"))
load(paste0(save_path, "/TCGA-PANCAN-EXPRESSION.RData"))
load(paste0(save_path, "/TCGA-PANCAN_MUTATION_index.RData"))
load(paste0(save_path, "/TCGA-PANCAN_MUTATION.RData"))
load(paste0(save_path, "/TCGA-PANCAN_CNA_index.RData"))
load(paste0(save_path, "/TCGA-PANCAN_CNA.RData"))
load(paste0(save_path, "/TCGA-PANCAN_METHYLATION_index.RData"))
load(paste0(save_path, "/TCGA-PANCAN_METHYLATION.RData"))

# CCLE-COSMIC RData load
load(paste0(save_path, "/CCLE-COSMIC-EXPRESSION.RData"))
load(paste0(save_path, "/CCLE-COSMIC-MUTATION.RData"))
load(paste0(save_path, "/CCLE-COSMIC-CNA.RData"))
load(paste0(save_path, "/CCLE-COSMIC-METHYLATION.RData"))

#
gene_expression_gene <- list(CCLE_EXP = ccle_exp_com$Gene, TCGA_EXP = tcga_exp_index$Gene)
ccle_tcga_gene <- gene_expression_gene %>% purrr::reduce(., intersect) %>% unique()

mutation_gene <- list(CCLE_MUT = ccle_mut_com$Gene, TCGA_MUT = tcga_mut_index$Gene)
ccle_tcga_mut <- mutation_gene %>% purrr::reduce(., intersect) %>% unique()

meth_probe <- list(CCLE_METH = ccle_meth_com$Probe, TCGA_METH = tcga_meth_index$Probe)
ccle_tcga_meth <- meth_probe %>% purrr::reduce(., intersect) %>% unique()

# cna_index <- .PrepCNA_custom(cna.original = ccle_cna_com)

venn_diagram(gene_expression_gene, type = "EXP")
venn_diagram(mutation_gene, type = "MUT")
venn_diagram(meth_probe, type = "METH")

exp.index <- tcga_exp_index %>% 
  filter(Gene %in% ccle_tcga_gene)
mut.index <- tcga_mut_index %>% 
  filter(Gene %in% ccle_tcga_mut)
meth.index <- tcga_meth_index %>% 
  filter(Probe %in% ccle_tcga_meth)

# index
save(exp.index, file = "TCGA_INDEX/CUSTOM/ccle_exp_custom_5454.RData")
save(mut.index, file = "TCGA_INDEX/CUSTOM/ccle_mut_custom_4946.RData")
save(meth.index, file = "TCGA_INDEX/CUSTOM/ccle_meth_custom_6231.RData")

# txt file
tcga_exp_ccl_paired <- exp.index %>% 
  select(Gene) %>% 
  left_join(x = ., y = tcga_exp_convert, by = "Gene")
tcga_exp_ccl_paired %>% 
  data.table::fwrite(file = "TCGA_INDEX/CUSTOM/tcga_exp_data_paired_with_ccl_custom.txt", sep = "\t")

tcga_mut_ccl_paired <- mut.index %>% 
  select(Gene) %>% 
  left_join(x = ., y = mut_longtowide, by = c("Gene" = "Hugo_Symbol"))
tcga_mut_ccl_paired %>% 
  data.table::fwrite(file = "TCGA_INDEX/CUSTOM/tcga_mut_data_paired_with_ccl_custom.txt", sep = "\t")

tcga_cna_ccl_paired <- tcga_cna_index %>% 
  select(CNA) %>% 
  left_join(x = ., y = tcga_cna_convert, by = "CNA")
tcga_cna_ccl_paired %>% 
  data.table::fwrite(file = "TCGA_INDEX/CUSTOM/tcga_cna_data_paired_with_ccl_custom.txt", sep = "\t")

tcga_meth_paired <- meth.index %>% 
  select(Probe) %>% 
  left_join(x = ., y = tcga_meth_convert, by = "Probe")
tcga_meth_paired %>% 
  data.table::fwrite(file = "TCGA_INDEX/CUSTOM/tcga_meth_data_paired_with_ccl_custom.txt", sep = "\t")

# TCGA-CCLE venn diagram
gene_expression_gene <- list(CCLE_EXP = ccle_exp_com$Gene, TCGA_EXP = tcga_exp_index$Gene)
mutation_gene <- list(CCLE_MUT = ccle_mut_com$Gene, TCGA_MUT = tcga_mut_index$Gene)
meth_probe <- list(CCLE_METH = ccle_meth_com$Probe, TCGA_METH = tcga_meth_index$Probe)

venn_diagram(gene_expression_gene, type = "EXP")
venn_diagram(mutation_gene, type = "MUT")
venn_diagram(meth_probe, type = "METH")

