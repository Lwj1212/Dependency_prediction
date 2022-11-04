source("~/WORK/gitworking/Dependency_prediction/preprocessing/src/function.R")

# train data
load("TRAIN/2022-08-18/CCLE-COSMIC-EXPRESSION.RData")
load("TRAIN/2022-08-18/CCLE-COSMIC-MUTATION.RData")
load("TRAIN/2022-08-18/CCLE-COSMIC-CNA.RData")
load("TRAIN/2022-08-18/CCLE-COSMIC-METHYLATION.RData")
load("TRAIN/2022-08-18/CCLE-COSMIC-GENEDEPENDENCY.RData")

path <- "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/DATA"
now_date <- Sys.time() %>% str_split(" ") %>% unlist() %>% .[1]
save_path <- paste0(path, "/", now_date)
dir.create(save_path, showWarnings = FALSE, recursive = TRUE)

Prep4DeepDEP_custom(
  # exp.data = ccle_exp_com ,
  mut.data = ccle_mut_com,
  # meth.data = ccle_meth_com,
  # cna.data = ccle_cna_com,
  dep.data = ccle_gene_dependency_com,
  mode = "training",
  filename.out = paste0(save_path, "/training")
)
