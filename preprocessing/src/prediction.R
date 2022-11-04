source("src/function.R")

# variable
path <- "/home/wmbio/WORK/gitworking/Dependency_prediction//preprocessing/PREDICTION"
now_date <- Sys.time() %>% str_split(" ") %>% unlist() %>% .[1]
save_path <- paste0(path, "/", now_date)
dir.create(save_path, showWarnings = TRUE, recursive = TRUE)

prep_path <- paste0(save_path, "/prep/")
dir.create(prep_path, showWarnings = TRUE, recursive = TRUE)

prep_com_path <- paste0(save_path, "/prep_com/")
dir.create(prep_com_path, showWarnings = TRUE, recursive = TRUE)


ccls_wmbio <- read_delim("WMBIO_CCLS.txt", delim = "\t", col_names = F) %>% pull(1)
ccle_omics_extraction(ccls = cclw_wmbio,
                      CCLE_SAMPLE_INFO = "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/RAW/CCLs/sample_info.csv", 
                      CCLE_EXP_PATH = "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/RAW/CCLs/CCLE_expression.csv",
                      CCLE_MUT_PATH = "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/RAW/CCLs/CCLE_mutations.csv",
                      CCLE_METH_PATH = "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/RAW/CCLs/CCLs_methylation_GSE68379.Rds",
                      CCLE_CNA_PATH = "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/RAW/CCLs/CCLE_segment_cn.csv",
                      save_path = prep_path)

# dep.data
# load("DepOI/DepOI-1632.RData")
load("DepOI/DepOI-1298-default.RData")

# Prep4DeepDEP 
type_list <- list.files(prep_path) %>% 
  lapply(X = ., FUN = function(value){
    tmp <- str_split(value, '_') %>% unlist() %>% .[1] %>% 
      str_split(., "-") %>% unlist() %>% sort()
    paste0(tmp, collapse = "-")
    
  }) %>% 
  unlist() %>% 
  unique()

for(tl in type_list){
  # meth-only or cna-only pass
  tryCatch(
    expr = {exp.data <- read.delim(file = paste0(prep_path, tl, "_prep_exp.txt"))},
    error = function(e){exp.data <<- NULL}
  )
  tryCatch(
    expr = {mut.data <- read.delim(file = paste0(prep_path, tl, "_prep_mut.txt"))},
    error = function(e){mut.data <<- NULL}
  )
  tryCatch(
    expr = {cna.data <- read.delim(file = paste0(prep_path, tl, "_prep_cna.txt"))},
    error = function(e){cna.data <<- NULL}
  )
  tryCatch(
    expr = {meth.data <- read.delim(file = paste0(prep_path, tl, "_prep_meth.txt"))},
    error = function(e){meth.data <<- NULL}
  )
  
  if((!is.null(exp.data) && ncol(exp.data) > 2) |
     (!is.null(mut.data) && ncol(mut.data) > 2) |
     (!is.null(meth.data) && ncol(meth.data) > 2) |
     (!is.null(cna.data) && length(unique(cna.data$CCLE_name)) > 1)){
    # predict
    Prep4DeepDEP(
      dep.data = dep.data,
      exp.data = exp.data,
      mut.data = mut.data,
      meth.data = meth.data,
      cna.data = cna.data,
      mode = "prediction",
      filename.out = paste0(prep_com_path, "/", tl, "_wmbio_ccls"))
  }
}


