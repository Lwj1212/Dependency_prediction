# library loading
{
  library(R.utils)
  library(biomaRt)
  library(tidyverse)
  library(GEOquery)
  library(minfi)
  library(RMariaDB)
  library(progress)
  library(Prep4DeepDEP)
}

# row to col convert
col_gene_row_cell <- function(DF, col_name = TRUE){
  if(col_name)
    DF <- DF %>% 
      dplyr::rename(DepMap_ID = V1)
  
  ## gene expression
  gene_name <- DF %>% 
    colnames() %>% 
    tibble(GENE = .) %>% 
    .[-1, ] %>% 
    separate(col = GENE, into = c("GENE", "tmp"), sep = " ") %>% 
    select(GENE) %>% 
    mutate(GENE = str_remove_all(string = GENE, pattern = "[:blank:]"))
  
  DF_t <- data.table::transpose(DF) %>% as_tibble()
  colnames(DF_t) <- DF_t[1,]
  DF_t <- DF_t[-1, ] %>% 
    dplyr::mutate_all(as.numeric) %>% 
    bind_cols(gene_name, .)
  
  return(DF_t)
  
}

# preprocessing
tcga_preprocessing <- function(save_path = "."){
  # setwd(save_path)
  
  # raw data load
  {
    tcga_exp_raw <- data.table::fread(file = "RAW/PANCANCER/tcga_RSEM_gene_tpm.gz") %>% 
      as_tibble()
    
    variant_type <- c("Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site")
    tcga_mut_raw <- data.table::fread(file = "RAW/PANCANCER/mc3.v0.2.8.PUBLIC.maf.gz") %>% 
      as_tibble() %>% 
      filter(Variant_Classification %in% variant_type) %>% 
      separate(col = Tumor_Sample_Barcode, into = LETTERS[1:7], sep = "-") %>% 
      select(-E,-F,-G) %>% 
      unite(col = "Tumor_Sample_Barcode", A,B,C,D, sep = "-") %>% 
      filter(str_detect(string = Tumor_Sample_Barcode, pattern = "^*A$")) %>% 
      mutate(Tumor_Sample_Barcode =  substring(Tumor_Sample_Barcode, 1, nchar(Tumor_Sample_Barcode) - 1))
    
    tcga_cna_raw <- data.table::fread(file = "RAW/PANCANCER/broad.mit.edu_PANCAN_Genome_Wide_SNP_6_whitelisted.seg") %>% 
      as_tibble() %>% 
      dplyr::rename(Tumor_Sample_Barcode = Sample) %>% 
      separate(col = Tumor_Sample_Barcode, into = LETTERS[1:7], sep = "-") %>% 
      select(-E,-F,-G) %>% 
      unite(col = "Tumor_Sample_Barcode", A,B,C,D, sep = "-") %>% 
      filter(str_detect(string = Tumor_Sample_Barcode, pattern = "^*A$")) %>% 
      mutate(Tumor_Sample_Barcode =  substring(Tumor_Sample_Barcode, 1, nchar(Tumor_Sample_Barcode) - 1))
    
    tcga_meth_raw <- data.table::fread(file = "RAW/PANCANCER/jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv") %>% 
      as_tibble() %>% 
      dplyr::rename(Probe = `Composite Element REF`)
    
    tcga_meth_sample <- tcga_meth_raw %>% colnames() %>% .[-1] %>% 
      lapply(X = ., FUN = function(s){
        tmp <- s %>% str_split(pattern = "-") %>% unlist() %>% 
          .[1:4] %>% 
          paste0(collapse = "-") %>% 
          substring(., 1, nchar(.) - 1)
        return(tmp)
      }) %>% unlist() %>% unique()
    colnames(tcga_meth_raw) <- c("Probe", tcga_meth_sample)
    
  }
  
  # omics sample extraction
  if(!file.exists(paste0(save_path, "/TCGA_OMICS_INTER_SAMPLE_BARCODE.txt"))){
    tcga_primary_sample <- read_delim(file = "RAW/PANCANCER/TCGA_phenotype_denseDataOnlyDownload.tsv",
                                      delim = "\t", show_col_types = FALSE) %>% 
      filter(sample_type == "Primary Tumor") %>% 
      pull(sample)
    
    ## exp
    tcga_exp_sample <- tcga_exp_raw %>% 
      as_tibble() %>% 
      colnames() %>% unique()
    
    ## mut
    tcga_mut_sample <- tcga_mut_raw %>% 
      pull(Tumor_Sample_Barcode) %>% 
      unique()
    
    # cna
    tcga_cna_sample <- tcga_cna_raw %>% 
      pull(Tumor_Sample_Barcode) %>% 
      unique() 
    
    # methlation
    tcga_meth_sample <- tcga_meth_raw %>% 
      colnames() %>% unique()
    
    tcga_omics_sample <- intersect(tcga_exp_sample, tcga_mut_sample) %>% 
      intersect(., tcga_cna_sample) %>% 
      intersect(., tcga_meth_sample)
    
    p_tcga <- ggVennDiagram::ggVennDiagram(x = list(Exp = tcga_exp_sample,
                                                    Mutation = tcga_mut_sample,
                                                    Cna = tcga_cna_sample,
                                                    Meth = tcga_meth_sample)) +
      ggtitle("TCGA-Omics integration") +
      scale_fill_gradient(low = "#F4FAFE", high = "#4981BF") +
      theme(legend.position = "none")
    ggsave(p_tcga, filename = paste0(save_path, "/TCGA-Omics_integration.png"), dpi = 200, width = 20, height = 10)
    tcga_omics_sample %>% tibble(Tumor_Sample_Barcode = .) %>% 
      write_delim(file = paste0(save_path, "/TCGA_OMICS_INTER_SAMPLE_BARCODE.txt"), delim = '\t')
    
  } else {
    tcga_omics_sample <- read_delim(file = paste0(save_path, "/TCGA_OMICS_INTER_SAMPLE_BARCODE.txt"), 
                                    delim = "\t", show_col_types = FALSE) %>% pull(1)
  }
  
  ## exp
  tcga_exp <- tcga_exp_raw %>%  
    dplyr::select(sample, all_of(tcga_omics_sample)) %>%
    mutate_if(is.numeric, function(value){
      normal <- 2^(value) - 0.001
      normal <- ifelse(normal < 0, 0, normal)
      log2(normal + 1) %>% return()
    }) %>% 
    separate(col = sample, into = c("Gene", "version"), sep = "\\.") %>% 
    dplyr::select(-version)
  
  mart <- useDataset("hsapiens_gene_ensembl", useMart("ensembl"))
  ensembl_id <- tcga_exp %>% 
    dplyr::select(Gene) %>% 
    getBM(filters= "ensembl_gene_id", attributes= c("ensembl_gene_id","hgnc_symbol"),
          values = .$Gene, mart= mart)
  
  tcga_exp_convert <- left_join(x = ensembl_id, y = tcga_exp , by = c("ensembl_gene_id" = "Gene")) %>% 
    dplyr::select(-ensembl_gene_id) %>% 
    dplyr::rename(Gene = hgnc_symbol) %>%
    filter(Gene != "") %>% 
    as_tibble()
  
  exp_sd <- apply(X = tcga_exp_convert[, -1], FUN = sd, MARGIN = 1, na.rm = TRUE) %>% tibble(exp_sd = .)
  exp_mean <- apply(X = tcga_exp_convert[, -1], FUN = mean, MARGIN = 1, na.rm = TRUE) %>% tibble(exp_mean = .)
  exp_sd_mean <- bind_cols(tcga_exp_convert %>% select_at(1), exp_mean, exp_sd)
  
  tcga_exp_index <- exp_sd_mean %>% 
    filter(exp_mean > 1 & exp_sd > 1) %>% 
    distinct(Gene, .keep_all = TRUE) %>%
    dplyr::select(Gene, Mean = exp_mean) %>%
    arrange(Gene)
  
  save(tcga_exp_convert, file = paste0(save_path, "/TCGA-PANCAN-EXPRESSION.RData"))
  save(tcga_exp_index, file = paste0(save_path, "/TCGA-PANCAN-EXPRESSION_index.RData"))
  
  rm(tcga_exp_raw, tcga_exp, tcga_exp_convert);gc()
  
  ## mut
  tcga_mut <- tcga_mut_raw %>% 
    filter(Tumor_Sample_Barcode %in% tcga_omics_sample) %>% 
    group_by(Tumor_Sample_Barcode, Hugo_Symbol) %>% 
    summarise(cnt = n()) %>% 
    mutate(cnt = 1)
  
  tcga_samples <- tcga_mut %>% 
    pull(Tumor_Sample_Barcode) %>% 
    unique()
  
  mut_longtowide <- lapply(X = tcga_samples, FUN = function(id){
    longtowide <- tcga_mut %>% 
      filter(Tumor_Sample_Barcode == id) %>%
      pivot_wider(names_from = "Tumor_Sample_Barcode", values_from = "cnt")
  }) %>% 
    purrr::reduce(.x = ., full_join, by = "Hugo_Symbol") %>% 
    replace(is.na(.), 0)
  
  # mutation frequency
  tcga_mut_convert <- mut_longtowide %>% 
    mutate(MUT_TRUE = rowSums( . == 1), MUT_FALSE = rowSums(. == 0)) %>% 
    mutate(MUT_FREQ = MUT_TRUE / (MUT_TRUE + MUT_FALSE)) %>% 
    dplyr::select(Hugo_Symbol, MUT_FREQ) 
  tcga_mut_convert$MUT_MEDIAN <- apply(mut_longtowide[, -1],1,function(v) median(as.numeric(v), na.rm = T))
  
  tcga_mut_index <- tcga_mut_convert %>%
    filter(MUT_FREQ >= 0.01) %>% 
    select(Gene = Hugo_Symbol, Median = MUT_MEDIAN)
  
  save(mut_longtowide, file = paste0(save_path, "/TCGA-PANCAN_MUTATION.RData"))
  # load("RData/TCGA-PANCAN_MUTATION.RData")
  save(tcga_mut_index, file = paste0(save_path, "/TCGA-PANCAN_MUTATION_index.RData"))
  
  rm(tcga_mut_raw, tcga_mut, mut_longtowide, tcga_mut_convert);gc()
  
  
  # Methylation
  tcga_meth_convert <- tcga_meth_raw %>% 
    dplyr::select(Probe, all_of(tcga_omics_sample)) %>% 
    na.omit()
  
  tcga_meth_index <- tcga_meth_convert %>% 
    mutate_if(is.numeric, .funs = function(col){
      ifelse(col <= 0.3, 1, 0) %>% return()
    }) %>% 
    mutate(METH_TRUE = rowSums( . == 1, na.rm = TRUE), METH_FALSE = rowSums(. == 0, na.rm = TRUE)) %>% 
    mutate(METH_FREQ = METH_TRUE / (METH_TRUE + METH_FALSE)) %>% 
    select(Probe, METH_FREQ) %>% 
    bind_cols(., apply(X = tcga_meth_convert[, -1], FUN = mean, MARGIN = 1, na.rm = TRUE) %>% 
                tibble(Mean = .)) %>%
    filter(METH_FREQ < 0.9) %>% 
    select(Probe, Mean)
  
  save(tcga_meth_convert, file = paste0(save_path,"/TCGA-PANCAN_METHYLATION.RData"))
  save(tcga_meth_index, file = paste0(save_path, "/TCGA-PANCAN_METHYLATION_index.RData"))
  
  rm(tcga_meth_raw, tcga_meth_convert);gc()
  
  # CNA
  tcga_cna <- tcga_cna_raw %>% 
    filter(Tumor_Sample_Barcode %in% tcga_omics_sample)
  
  #  prep CNA function
  tcga_sample <- unique(tcga_cna$Tumor_Sample_Barcode)
  if (sum(tolower(colnames(tcga_cna)) %in% c("chr", "chromosome")) == 1) {
    col_idx <- which(tolower(colnames(tcga_cna)) %in% c("chr", "chromosome"))
    tcga_cna[which(tolower(tcga_cna[, col_idx]) == "x"), col_idx] <- 23
    tcga_cna[which(tolower(tcga_cna[, col_idx]) == "y"), col_idx] <- 24
    colnames(tcga_cna)[col_idx] <- "Chromosome"
  }
  
  dir.create(paste0(save_path, "/TCGA-PANCAN-CNV"), showWarnings = FALSE)
  cnv_list <- list.files(paste0(save_path, '/TCGA-PANCAN-CNV'), full.names = T)
  if(length(cnv_list) >= 22){
    bigTable_list <- list()
    for(index in 1:length(cnv_list)){
      print(cnv_list[index])
      load(cnv_list[index])
      bigTable_list[[index]] <- sample_bigTable %>% 
        lapply(X = ., FUN = function(col){ 
          col %>% select(-CNA)}) %>% 
        bind_cols(sample_bigTable[[1]] %>% select(CNA), .)
    }
  } else {
    BP_SIZE <- 10 ^ 5
    chr <- unique(tcga_cna[, col_idx]) %>% pull()
    chrom_length <- c(249260000, 243200000, 198030000, 191160000, 
                      180920000, 171120000, 159140000, 146370000, 141220000, 
                      135540000, 135010000, 133860000, 115170000, 107350000, 
                      102540000, 90360000, 81200000, 78080000, 59130000, 63030000, 
                      48130000, 51310000, 155280000, 59380000)  ## chromomse bp length
    chrom_bin <- ceiling(chrom_length/BP_SIZE)
    bigTable <- data.frame(matrix(data = 0, ncol = length(tcga_sample) +  1, nrow = sum(chrom_bin)), 
                           stringsAsFactors = FALSE)
    colnames(bigTable) <- c("CNA", tcga_sample)
    
    k = 1
    for (i in 1:length(chr)) {
      bin_start <- seq(0, chrom_bin[i] - 1, 1)
      bin_end <- seq(1, chrom_bin[i], 1)
      bigTable$CNA[k:(k + chrom_bin[i] - 1)] <- paste(paste0("chr", i), paste0(bin_start, "to", bin_end), "100kb", sep = "_")
      k = chrom_bin[i] + k
    }
    # not valid chr
    bigTable <- bigTable %>% 
      filter(CNA != "0")
    
    bigTable_Start_End <- bigTable %>% select(CNA) %>% 
      separate(col = "CNA", into = c("Chr", "Start_End", "bin"), sep = "_", remove = FALSE) %>% 
      select(-bin) %>% 
      separate(col = "Start_End", into = c("Start", "End"), sep = "to") %>% 
      mutate(Start = as.numeric(Start), End = as.numeric(End))
    
    bigTable_list <- list()
    for (chr_ in chr[1:length(chr)]) { ## Chromosome 별로
      bigTable_chr <- bigTable %>% 
        filter(str_detect(string = CNA, pattern = paste0("^chr", chr_, "_")))
      cna.filterTable <- bigTable_Start_End %>% 
        filter(Chr == paste0("chr",chr_))
      idx.chr <- which(tcga_cna$Chromosome == chr_)
      Table.chr <- tcga_cna[idx.chr, ]
      cna.length.chr <- (Table.chr$End - Table.chr$Start) + 1  ## Start - end
      
      print(paste0("Chromosome : ", chr_))
      
      sample_bigTable <- pbmcapply::pbmclapply(X = 2:ncol(bigTable_chr), FUN = function(j){ #ncol(bigTable_chr)
        idx.cell <- which(Table.chr$Tumor_Sample_Barcode == colnames(bigTable_chr)[j])
        idx.big <- which(colnames(bigTable_chr) == colnames(bigTable_chr)[j])
        cellTable.chr <- Table.chr[idx.cell, ]
        cna.length <- cna.length.chr[idx.cell]
        
        tmp_bigTable <- bigTable_chr %>% select_at(.vars = c(1,idx.big))
        gc()
        
        for (k in cna.filterTable$End) {
          end_matrix <- data.frame(matrix(data = 1, nrow = nrow(cellTable.chr)) * BP_SIZE * k, stringsAsFactors = FALSE)
          end_matrix$cellTable <- cellTable.chr$End
          start_matrix <- data.frame(matrix(data = 1, nrow = nrow(cellTable.chr)) * BP_SIZE * (k - 1) + 1, stringsAsFactors = FALSE)
          start_matrix$cellTable <- cellTable.chr$Start
          overlap.length <- (BP_SIZE) + cna.length - (apply(end_matrix, 1, max) - apply(start_matrix, 1, min) + 1)
          overlap.length[overlap.length < 0] <- 0
          tmp_bigTable[k, 2] <- round(sum(overlap.length * cellTable.chr$Segment_Mean)/BP_SIZE, digits = 4)
        }
        
        return(tmp_bigTable)
      }, mc.cores = 24)
      
      # join
      save(sample_bigTable, file = paste0(save_path, '/TCGA-PANCAN-CNV/chr', chr_, ".RData"))
      
      print("sample bigTable join...")
      bigTable_list[[chr_]] <- sample_bigTable %>% 
        lapply(X = ., FUN = function(col){ 
          col %>% select(-CNA)}) %>% 
        bind_cols(sample_bigTable[[1]] %>% select(CNA), .)
      print("sample bigTable join... : Done!")
      print(paste0("Chromosome : ", chr_, " Done!"))
    }
  }
  
  # all sample CNA
  tcga_cna_convert <- bigTable_list %>% 
    bind_rows() %>% 
    separate(col = CNA, into = LETTERS[1:3], sep = "_", remove = FALSE) %>% 
    select(-B,-C) %>% 
    mutate(CHR = as.numeric(str_remove_all(A, "chr"))) %>% 
    select(-A) %>% 
    arrange(CHR) %>% 
    select(-CHR)
  
  # filter
  bigTable_com_zeros <- tcga_cna_convert %>% 
    mutate(CNA_NON_ZEROS = rowSums(. != 0, na.rm = TRUE), CNA_ZEROS = rowSums(. == 0, na.rm = TRUE)) %>% 
    mutate(CNA_ZEROS_FREQ = CNA_ZEROS / (CNA_NON_ZEROS + CNA_ZEROS)) %>% 
    select(CNA_ZEROS_FREQ)
  cv <- apply(X = tcga_cna_convert[,-1], MARGIN = 1, function(x)sd(x, na.rm = TRUE) / mean(x, na.rm = TRUE) * 100) %>% 
    tibble(CV = .)
  absmean <- apply(X = tcga_cna_convert[,-1], MARGIN = 1, function(x) mean(abs(x), na.rm = TRUE)) %>% 
    tibble(ABS_MEAN = .)
  
  tcga_cna_index <- tcga_cna_convert %>% 
    select(CNA) %>% 
    bind_cols(., bigTable_com_zeros, cv, absmean) %>% 
    filter(CNA_ZEROS_FREQ <= 0.05 & CV > 0.2 & ABS_MEAN > 0.15)
  
  tcga_cna_index <-  tcga_cna_index %>% 
    select(CNA) %>% 
    separate(col = CNA, into = c("Chr", "Start_END", "REMOVE1"), sep = "_", remove = FALSE) %>% 
    select(-REMOVE1) %>% 
    separate(col = Start_END, into = c("Start", "End"), sep = "to") %>% 
    mutate(Chr = as.numeric(str_remove_all(Chr, "chr")), Start = as.numeric(Start), End = as.numeric(End))
  
  save(tcga_cna_convert, file = paste0(save_path, "/TCGA-PANCAN_CNA.RData"))
  save(tcga_cna_index, file = paste0(save_path, "/TCGA-PANCAN_CNA_index.RData"))
  
}
ccle_preprocessing <- function(save_path = ".", 
                               CCLE_SAMPLE_INFO = "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/RAW/CCLs/sample_info.csv",
                               CCLE_EXP_PATH = "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/RAW/CCLs/CCLE_expression.csv",
                               CCLE_MUT_PATH = "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/RAW/CCLs/CCLE_mutations.csv",
                               CCLE_METH_PATH = "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/RAW/CCLs/CCLs_methylation_GSE68379.Rds",
                               CCLE_CNA_PATH = "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/RAW/CCLs/CCLE_segment_cn.csv",
                               CCLE_GD_PATH = "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/RAW/CCLs/CRISPR_gene_effect.csv",
                               remove_default_OI = FALSE,
                               DepOI_SD = NULL){
  setwd(path)
  
  sample_info <-  data.table::fread(CCLE_SAMPLE_INFO) %>% 
    as_tibble() %>% 
    select(DepMap_ID, stripped_cell_line_name, COSMICID) %>%
    filter(!is.na(COSMICID)) %>% 
    arrange(DepMap_ID)
  
  # gene expression
  ccle_exp <- data.table::fread(CCLE_EXP_PATH) %>% 
    as_tibble() %>% 
    col_gene_row_cell()
  
  ccle_exp_col_name <- ccle_exp %>% 
    colnames() %>% 
    .[-1] %>% 
    tibble(DepMap_ID = .)
  
  ccle_exp_col_name_com <- left_join(x = ccle_exp_col_name, y = sample_info, by = "DepMap_ID") %>% 
    pull(stripped_cell_line_name)
  colnames(ccle_exp) <- c("Gene", ccle_exp_col_name_com)
  
  # COSMIC ID exist
  ccle_exp_cosmic <- which(!is.na(colnames(ccle_exp)), arr.ind = TRUE)
  ccle_exp <- ccle_exp %>% 
    select_at(ccle_exp_cosmic) %>% 
    distinct(Gene, .keep_all = TRUE) %>% 
    as.data.frame()
  
  
  ## mut - nonsense, missense, frameshit IS, DEL, splice
  mutation_type <- c('Nonsense_Mutation', 'Missense_Mutation', 'Frame_Shift_Del', 
                     'Frame_Shift_Ins','Splice_Site' )
  mut <- data.table::fread(file = CCLE_MUT_PATH) %>% 
    as_tibble() %>% 
    select(DepMap_ID, everything()) %>% 
    filter(Variant_Classification %in% mutation_type) %>% 
    group_by(DepMap_ID, Hugo_Symbol) %>% 
    summarise(cnt = n()) %>% 
    mutate(cnt = 1)
  
  ccls <- mut %>% 
    pull(DepMap_ID) %>% 
    unique()
  
  ccle_mut <- lapply(X = ccls, FUN = function(id){
    longtowide <- mut %>% 
      filter(DepMap_ID == id) %>%
      pivot_wider(names_from = "DepMap_ID", values_from = "cnt")
  }) %>% 
    purrr::reduce(.x = ., full_join, by = "Hugo_Symbol") %>% 
    replace(is.na(.), 0)
  
  ccle_mut_col_name <- ccle_mut %>% 
    colnames() %>% 
    .[-1] %>% 
    tibble(DepMap_ID = .)
  ccle_mut_col_name_com <- left_join(x = ccle_mut_col_name, y = sample_info, by = "DepMap_ID") %>% 
    pull(stripped_cell_line_name)
  colnames(ccle_mut) <- c("Gene", ccle_mut_col_name_com)
  
  # COSMIC ID exist
  ccle_mut_cosmic <- which(!is.na(colnames(ccle_mut)), arr.ind = TRUE)
  ccle_mut <- ccle_mut %>% select_at(ccle_mut_cosmic) %>% 
    distinct(Gene, .keep_all = TRUE) %>% 
    as.data.frame()
  
  ## CNA
  ccle_cna <- data.table::fread(file = CCLE_CNA_PATH) %>% 
    as_tibble() %>% 
    select(-Status, -Source) %>% 
    select(CCLE_name = DepMap_ID, Chromosome, Start, End, Num_Probes, Segment_Mean) %>% 
    left_join(x = ., y = sample_info, by = c("CCLE_name" = "DepMap_ID")) %>% 
    select(-CCLE_name, -COSMICID) %>% 
    select(CCLE_name = stripped_cell_line_name, everything()) %>% 
    filter(!is.na(CCLE_name)) %>% 
    as.data.frame()
  
  
  # methylation
  #increase file download timeout
  if(!file.exists(CCLE_METH_PATH)){
    #get raw data - idats, processed beta matrix, etc.
    getGEOSuppFiles("GSE68379")
    
    #list files
    idatFiles <- list.files("RAW/CCLs/CCLs_methylation_GSE68379", pattern = "idat.gz$", full = TRUE)
    
    #decompress individual idat files
    sapply(idatFiles, gunzip, overwrite = TRUE)
    
    #read idats and create RGSet
    RGSet <- read.metharray.exp("RAW/CCLs//CCLs_methylation_GSE68379")
    saveRDS(RGSet, "RAW/CCLs/CCLs_methylation_GSE68379.Rds")
  } else {
    RGSet <- readRDS(CCLE_METH_PATH)
  }
  
  # phenotype
  geoMat <- getGEO("GSE68379")
  pD.all <- pData(geoMat[[1]])
  col_name <- pD.all$characteristics_ch1.3
  
  # get methylation beta
  grSet <- preprocessQuantile(RGSet)
  grBeta <- getBeta(grSet) %>% 
    as.data.frame()
  colnames(grBeta) <- col_name
  grBeta_NULL <- which(colnames(grBeta) == "", arr.ind = TRUE)
  ccle_meth <- grBeta[, -grBeta_NULL] %>% 
    rownames_to_column(var = "Probe")
  
  # methylation
  ccle_meth_col_name <- ccle_meth %>% 
    colnames() %>% 
    lapply(X = ., FUN = function(value){
      str_split(string = value, pattern = " ") %>% 
        unlist() %>% .[2] %>% 
        return()
    }) %>% unlist() %>% 
    tibble(COSMICID = .) %>% 
    mutate(COSMICID = as.integer(COSMICID)) %>%  .[-1, ]
  
  ccle_meth_col_name_com <- left_join(x = ccle_meth_col_name, y = sample_info, by = "COSMICID") %>%
    distinct(COSMICID, .keep_all = TRUE) %>% 
    pull(stripped_cell_line_name)
  colnames(ccle_meth) <- c("Probe", ccle_meth_col_name_com)
  
  # COSMIC ID exist
  ccle_meth_cosmic <- which(!is.na(colnames(ccle_meth)), arr.ind = TRUE)
  ccle_meth <- ccle_meth %>% select_at(ccle_meth_cosmic) %>% 
    distinct(Probe, .keep_all = TRUE) %>% 
    as.data.frame()
  
  # gene dependency
  ccle_gene_dependency <- data.table::fread(file = CCLE_GD_PATH) %>%
    as_tibble() %>% 
    col_gene_row_cell(col_name = F)
  
  ccle_gene_dependency_col_name <- ccle_gene_dependency %>% 
    colnames() %>% 
    .[-1] %>% 
    tibble(DepMap_ID = .)
  ccle_gene_dependency_col_name_com <- left_join(x = ccle_gene_dependency_col_name, y = sample_info, by = "DepMap_ID") %>% 
    pull(stripped_cell_line_name)
  colnames(ccle_gene_dependency) <- c("Gene", ccle_gene_dependency_col_name_com)
  
  # COSMIC ID exist
  ccle_gene_dependency_cosmic <- which(!is.na(colnames(ccle_gene_dependency)), arr.ind = TRUE)
  ccle_gene_dependency <- ccle_gene_dependency %>% select_at(ccle_gene_dependency_cosmic) %>% 
    distinct(Gene, .keep_all = TRUE) %>% 
    as.data.frame()
  
  # CCLE omics integration venn diagram
  omics_ccl_list <- list(
    GeneDependency = ccle_gene_dependency %>% colnames() %>% .[-1],
    Exp = ccle_exp %>% colnames() %>% .[-1],
    Mutation = ccle_mut %>% colnames() %>% .[-1],
    Cna = ccle_cna %>% pull(CCLE_name) %>% unique(),
    Methylation = ccle_meth %>% colnames() %>% .[-1])
  ccle_omics_intersection <- purrr::reduce(omics_ccl_list, intersect)
  
  p_v <- ggVennDiagram::ggVennDiagram(x = omics_ccl_list) +
    ggtitle("CCLE-Omics integration") + 
    scale_fill_gradient(low = "#F4FAFE", high = "#4981BF") +
    theme(legend.position = "none")
  ggsave(p_v, filename = paste0(save_path, "/CCLE-Omics_integration.png"), dpi = 200, width = 20, height = 10)
  
  ccle_exp_com <- ccle_exp %>% select(Gene, any_of(ccle_omics_intersection))
  ccle_mut_com <- ccle_mut %>% select(Gene, any_of(ccle_omics_intersection))
  ccle_cna_com <- ccle_cna %>% filter(CCLE_name %in% ccle_omics_intersection)
  ccle_meth_com <- ccle_meth %>% select(Probe, any_of(ccle_omics_intersection))
  ccle_gene_dependency <- ccle_gene_dependency %>% select(Gene, any_of(ccle_omics_intersection))
  
  # DepOIs
  if(is.null(DepOI_SD)){
    load(DepOI_defualt)
  } else {
    gd_matrix <- ccle_gene_dependency %>% select(-1) %>% as.matrix()
    rownames(gd_matrix) <- ccle_gene_dependency$Gene
    
    dep.data_custom <- apply(gd_matrix, MARGIN = 1, sd, na.rm = TRUE) %>% 
      tibble(Gene = rownames(gd_matrix), SD = .) %>% 
      filter(SD >= DepOI_SD) %>% 
      select(Gene)
    
    if(remove_default_OI){
      path <- system.file("extdata/", package = "Prep4DeepDEP")
      load(paste0(path, "default_dep_genes_1298.RData"))
      dep.data <- setdiff(dep.data_custom[[1]], dep.data[[1]]) %>% tibble(Gene = .)
    }
  }

  ccle_gene_dependency_com <- ccle_gene_dependency %>% 
    inner_join(x = ., y = dep.data, by = "Gene")
  
  # Save train dataset
  save(ccle_exp_com, file = paste0(save_path,"/CCLE-COSMIC-EXPRESSION.RData"))
  save(ccle_mut_com, file = paste0(save_path, "/CCLE-COSMIC-MUTATION.RData"))
  save(ccle_cna_com, file = paste0(save_path, "/CCLE-COSMIC-CNA.RData"))
  save(ccle_meth_com, file = paste0(save_path, "/CCLE-COSMIC-METHYLATION.RData"))
  save(ccle_gene_dependency_com, file = paste0(save_path, "/CCLE-COSMIC-GENEDEPENDENCY.RData"))
  
}

# prediction
ccle_omics_extraction <- function(ccls, CCLE_SAMPLE_INFO = "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/RAW/CCLs/sample_info.csv",
                                  CCLE_EXP_PATH = "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/RAW/CCLs/CCLE_expression.csv",
                                  CCLE_MUT_PATH = "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/RAW/CCLs/CCLE_mutations.csv",
                                  CCLE_METH_PATH = "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/RAW/CCLs/CCLs_methylation_GSE68379.Rds",
                                  CCLE_CNA_PATH = "/home/wmbio/WORK/gitworking/Dependency_prediction/preprocessing/RAW/CCLs/CCLE_segment_cn.csv",
                                  save_path = "."){
  
  # sample information
  sample_info <-  data.table::fread(CCLE_SAMPLE_INFO) %>% 
    as_tibble() %>% 
    select(DepMap_ID, stripped_cell_line_name, COSMICID) %>%
    # filter(!is.na(COSMICID)) %>% 
    arrange(DepMap_ID)
  
  # gene expression
  ccle_exp <- data.table::fread(CCLE_EXP_PATH) %>% 
    as_tibble() %>% 
    col_gene_row_cell() %>% 
    distinct(GENE, .keep_all = TRUE) %>% 
    as_tibble()
  
  ccle_exp_col_name  <- ccle_exp %>% 
    colnames() %>% 
    .[-1] %>% 
    tibble(DepMap_ID = .)
  
  ccle_exp_col_name_com <- left_join(x = ccle_exp_col_name, y = sample_info, by = "DepMap_ID") %>% 
    pull(stripped_cell_line_name)
  colnames(ccle_exp) <- c("Gene", ccle_exp_col_name_com)
  
  ## mut - nonsense, missense, frameshit IS, DEL, splice
  mutation_type <- c('Nonsense_Mutation', 'Missense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins','Splice_Site' )
  mut <- data.table::fread(file = CCLE_MUT_PATH) %>% 
    as_tibble() %>% 
    select(DepMap_ID, everything()) %>% 
    filter(Variant_Classification %in% mutation_type) %>% 
    group_by(DepMap_ID, Hugo_Symbol) %>% 
    summarise(cnt = n()) %>% 
    mutate(cnt = 1)
  
  ccls <- mut %>% 
    pull(DepMap_ID) %>% 
    unique()
  
  ccle_mut <- lapply(X = ccls, FUN = function(id){
    longtowide <- mut %>% 
      filter(DepMap_ID == id) %>%
      pivot_wider(names_from = "DepMap_ID", values_from = "cnt")
  }) %>% 
    purrr::reduce(.x = ., full_join, by = "Hugo_Symbol") %>% 
    replace(is.na(.), 0) %>% 
    as_tibble()
  
  ccle_mut_col_name <- ccle_mut %>% 
    colnames() %>% 
    .[-1] %>% 
    tibble(DepMap_ID = .)
  ccle_mut_col_name_com <- left_join(x = ccle_mut_col_name, y = sample_info, by = "DepMap_ID") %>% 
    pull(stripped_cell_line_name)
  colnames(ccle_mut) <- c("Gene", ccle_mut_col_name_com)
  
  # CNA
  ccle_cna <- data.table::fread(file = CCLE_CNA_PATH) %>% 
    as_tibble() %>% 
    select(-Status, -Source) %>% 
    select(CCLE_name = DepMap_ID, Chromosome, Start, End, Num_Probes, Segment_Mean) %>% 
    left_join(x = ., y = sample_info, by = c("CCLE_name" = "DepMap_ID")) %>% 
    select(-CCLE_name) %>%
    select(CCLE_name = stripped_cell_line_name, everything()) %>% 
    filter(!is.na(CCLE_name)) %>% 
    as_tibble()
  
  # Methylation
  if(!file.exists(CCLE_METH_PATH)){
    #get raw data - idats, processed beta matrix, etc.
    getGEOSuppFiles("GSE68379")
    
    #list files
    idatFiles <- list.files("RAW/CCLs/CCLs_methylation_GSE68379", pattern = "idat.gz$", full = TRUE)
    
    #decompress individual idat files
    sapply(idatFiles, gunzip, overwrite = TRUE)
    
    #read idats and create RGSet
    RGSet <- read.metharray.exp("RAW/CCLs//CCLs_methylation_GSE68379")
    saveRDS(RGSet, "RAW/CCLs/CCLs_methylation_GSE68379.Rds")
  } else {
    RGSet <- readRDS(CCLE_METH_PATH)
  }
  
  # phenotype
  geoMat <- getGEO("GSE68379")
  pD.all <- pData(geoMat[[1]])
  col_name <- pD.all$characteristics_ch1.3
  
  # get methylation beta
  grSet <- preprocessQuantile(RGSet)
  grBeta <- getBeta(grSet) %>% 
    as.data.frame()
  colnames(grBeta) <- col_name
  grBeta_NULL <- which(colnames(grBeta) == "", arr.ind = TRUE)
  ccle_meth <- grBeta[, -grBeta_NULL] %>% 
    rownames_to_column(var = "Probe")
  
  # methylation
  ccle_meth_col_name <- ccle_meth %>% 
    colnames() %>% 
    lapply(X = ., FUN = function(value){
      str_split(string = value, pattern = " ") %>% 
        unlist() %>% .[2] %>% 
        return()
    }) %>% unlist() %>% 
    tibble(COSMICID = .) %>% 
    mutate(COSMICID = as.integer(COSMICID)) %>%  .[-1, ]
  
  ccle_meth_col_name_com <- left_join(x = ccle_meth_col_name, y = sample_info, by = "COSMICID") %>%
    distinct(COSMICID, .keep_all = TRUE) %>% 
    pull(stripped_cell_line_name)
  colnames(ccle_meth) <- c("Probe", ccle_meth_col_name_com)
  
  # COSMIC ID exist
  ccle_meth_cosmic <- which(!is.na(colnames(ccle_meth)), arr.ind = TRUE)
  ccle_meth <- ccle_meth %>% select_at(ccle_meth_cosmic) %>% 
    distinct(Probe, .keep_all = TRUE) %>% 
    as_tibble()
  
  # intersection check
  omics_col_name <- list(exp = ccle_exp %>% select(any_of(ccls_wmbio)) %>% colnames(),
                         mut = ccle_mut %>% select(any_of(ccls_wmbio)) %>% colnames(),
                         cna = ccle_cna %>% filter(CCLE_name %in% ccls_wmbio) %>% .$CCLE_name %>% unique(),
                         meth = ccle_meth %>% select(any_of(ccls_wmbio)) %>% colnames())
  
  # omics intersect
  v <- venn_diagram(inter_list = omics_col_name)
  intersect_target <- v$layers[[1]]$data$item
  names(intersect_target) <- v$layers[[1]]$data$name
  
  tmp <- intersect_target[['exp..mut']]
  
  ccle_meth %>% select(Probe, any_of(tmp))
  ccle_cna %>% filter(CCLE_name %in% tmp)
  
  new_name <- names(intersect_target) %>% 
    lapply(X = ., FUN = function(n){
      n_split <- str_split(n, "\\..") %>% unlist() %>% 
        paste0(collapse = "-")
    }) %>% unlist()
  
  names(intersect_target) <- new_name
  
  lapply(X = names(intersect_target), FUN = function(value){
    target_list <- intersect_target[[value]]
    
    tmp <- str_split(value, '_') %>% unlist() %>% .[1] %>% 
      str_split(., "-") %>% unlist() %>% sort()
    value <- paste0(tmp, collapse = "-")
    
    if(length(target_list) <= 0)
      return(NULL)
    
    exp.data <- ccle_exp %>% select(Gene, any_of(target_list))
    mut.data <- ccle_mut %>% select(Gene, any_of(target_list))
    meth.data <- ccle_meth %>% select(Probe, any_of(target_list))
    cna.data <- ccle_cna %>% filter(CCLE_name %in% target_list)
    
    if(ncol(exp.data) > 1)
      write_delim(x = exp.data, file = paste0(save_path, "/", value, "_prep_exp.txt"), delim = "\t")
    if(ncol(mut.data) > 1)
      write_delim(x = mut.data, file = paste0(save_path, "/", value, "_prep_mut.txt"), delim = "\t")
    if(ncol(meth.data) > 1)
      write_delim(x = meth.data, file = paste0(save_path, "/", value, "_prep_meth.txt"), delim = "\t")
    if(nrow(cna.data) >= 1)
      write_delim(x = cna.data, file = paste0(save_path, "/", value, "_prep_cna.txt"), delim = "\t")
    
  })
}

# CCLE omics integration venn diagram
venn_diagram <- function(inter_list, save_path = ".", type = "", ggtitle_text = ""){
  p_v <- ggVennDiagram::ggVennDiagram(x = inter_list) +
    ggtitle(ggtitle_text) + 
    scale_fill_gradient(low = "#F4FAFE", high = "#4981BF") +
    theme(legend.position = "none")
  ggsave(p_v, filename = paste0(save_path, "/CCLE-TCGA_integration_", type, ".png"), dpi = 200, width = 20, height = 10)
  return(p_v)
}

Prep4DeepDEP_custom <- function (exp.data = NULL, mut.data = NULL, meth.data = NULL, cna.data = NULL, dep.data = NULL, 
                          mode = c("Training", "Prediction"), filename.out = "data"){
  
    check.cellNames <- NULL
    cat("Mode", mode, "\n\n")
    path <- system.file("extdata/", package = "Prep4DeepDEP")
    if (sum(is.null(exp.data), is.null(mut.data), is.null(meth.data), 
            is.null(cna.data)) == 4) {
      stop(c("All genomic profiles are missing. Please provide at least one of mut.data, exp.data, meth.data, and cna.data."))
    }
    if (is.null(dep.data) & tolower(mode) == "prediction") {
      cat("dep.data is not provided, running with the default 1298 DepOIs...", 
          "\n")
      load(paste0(path, "default_dep_genes_1298.RData"))
    }
    if (is.null(dep.data) & tolower(mode) == "training") {
      cat("dep.data is not provided. Please provide gene dependency scores for the training mode...", 
          "\n")
    }
    if (ncol(dep.data) == 1 & tolower(mode) == "training") {
      stop(c("Only one column detected in dep.data. Please provide gene dependency symbols and scores for the training mode."), 
           call. = FALSE)
    }
    load(paste0(path, "gene_fingerprints_CGP.RData"))
    list.genes <- .CheckGeneSymbol(dep.data = dep.data, filename.out = filename.out)
    n <- nrow(list.genes)
    
    # Gene expression
    if (!is.null(exp.data)) {
        if (!is.character(exp.data[1, 1])) {
            stop("exp.data format error, please check!", call. = FALSE)
        }
        cat(c("Exp started..."), "\n")
        colnames(exp.data)[1] <- "Gene"
        ncell <- ncol(exp.data[, -1])
        if (is.null(check.cellNames)) {
            check.cellNames <- colnames(exp.data[, -1])
        }else if (length(check.cellNames) != length(colnames(exp.data[, -1])) | sum(check.cellNames %in% colnames(exp.data[, -1])) != length(check.cellNames)) {
            stop(c("Cell line names are inconsistent!"), call. = FALSE)
        }
        cat("Precessing", paste(length(check.cellNames), "cell lines..."), "\n")
        inputData <- exp.data[!duplicated(exp.data$Gene), ]

        ######## TCGA - CCLE intersection        
        load(paste0(path, "ccle_exp_for_missing_value_6016.RData"))

        # outputData <- merge(exp.index, inputData, by = "Gene", sort = FALSE, all.x = TRUE)
        outputData <- merge(exp.index, inputData, by = "Gene", sort = FALSE, all.x = TRUE)

        Gene <- outputData$Gene
        rownames(outputData) <- outputData$Gene
        value_NA <- rowSums(outputData[, -c(1, 2)])
        cat(sum(is.na(value_NA)), "genes with NA values in exp.data. Substitute by mean values of CCLE.", "\n")
        if (round((sum(is.na(value_NA))/nrow(outputData)), digits = 2) > 0.2) {
            warning("NA found in >20% genes, please check if input format is correct!")
        }
        for (i in 1:nrow(outputData)) {
            if (is.na(value_NA[i])) {
                outputData[i, is.na(outputData[i, ])] <- outputData$Mean[i]
            }
        }
        outputData <- round(as.matrix(outputData[, -1]), digits = 4)
        outputData.final.exp <- cbind(Gene, as.data.frame(outputData[, -1], stringsAsFactors = FALSE))
        outputData.final.exp <- outputData.final.exp[, c("Gene", check.cellNames)]
        if (tolower(mode) == "prediction") {
            data.table::fwrite(outputData.final.exp, 
                               file = paste(filename.out, "exp_prediction.txt", sep = "_"), sep = "\t", 
                               row.names = FALSE, col.names = TRUE, quote = FALSE)
        }
        if (tolower(mode) == "training") {
          k = 2:ncol(outputData.final.exp)
          rep_col_list <- pbmcapply::pbmclapply(X = k, FUN = function(index){
            rep_col.1 <- do.call("cbind", replicate(n, outputData.final.exp[, index], simplify = FALSE)) # 
            colnames(rep_col.1) <- paste0("C", index-1, "G", seq(1, n, 1))
            return(rep_col.1)
          }, mc.cores = 4)
          
          rep_col <- rep_col_list %>% 
            bind_cols(tibble(Gene = rownames(outputData.final.exp)), .) 
            
          data.table::fwrite(rep_col, file = paste(filename.out, "exp_training.txt", sep = "_"), 
                      sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
        }
        rm(exp.data, inputData, outputData, outputData.final.exp, rep_col_list, rep_col);gc()
        cat("Exp completed!", "\n\n")
    }
    # Mutation
    if (!is.null(mut.data)) {
      if (!is.character(mut.data[1, 1])) {
        stop("mut.data format error, please check!", call. = FALSE)
      }
      cat(c("Mut started..."), "\n")
      colnames(mut.data)[1] <- "Gene"
      ncell <- ncol(mut.data[, -1])
      if (is.null(check.cellNames)) {
        check.cellNames <- colnames(mut.data[, -1])
      } else if (length(check.cellNames) != length(colnames(mut.data[, -1])) | sum(check.cellNames %in% colnames(mut.data[, -1])) != length(check.cellNames)) {
        stop(c("Cell line names are inconsistent!"), call. = FALSE)
      }
      cat("Precessing", paste(length(check.cellNames), "cell lines..."), "\n")
      inputData <- mut.data[!duplicated(mut.data$Gene), ]
      
      ######## TCGA - CCLE intersection        
      load(paste0(path, "ccle_mut_for_missing_value_4539.RData"))
      
      outputData <- merge(mut.index, inputData, by = "Gene", sort = FALSE, all.x = TRUE)
      Gene <- outputData$Gene
      rownames(outputData) <- outputData$Gene
      value_NA <- rowSums(outputData[, -c(1, 2)])
      cat(sum(is.na(value_NA)), "genes with NA values in mut.data. Substitute by median values of CCLE.", "\n")
      if (round((sum(is.na(value_NA))/nrow(outputData)), digits = 2) > 0.2) {
        warning("NA found in >20% genes, please check if input format is correct!")
      }
      for (i in 1:nrow(outputData)) {
        if (is.na(value_NA[i])) {
          outputData[i, is.na(outputData[i, ])] <- outputData$Median[i]
        }
      }
      outputData.final.mut <- cbind(Gene, as.data.frame(outputData[, -c(1, 2)]))
      outputData.final.mut <- outputData.final.mut[, c("Gene", check.cellNames)]
      if (tolower(mode) == "prediction") {
        data.table::fwrite(outputData.final.mut, 
                           file = paste(filename.out, "mut_prediction.txt", sep = "_"), 
                           sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
      }
      if (tolower(mode) == "training") {
        k = 2:ncol(outputData.final.mut)
        rep_col_list <- pbmcapply::pbmclapply(X = k, FUN = function(index){
          rep_col.1 <- do.call("cbind", replicate(n, outputData.final.mut[, index], simplify = FALSE)) # 
          colnames(rep_col.1) <- paste0("C", index-1, "G", seq(1, n, 1))
          return(rep_col.1)
        }, mc.cores = 4)
        
        rep_col <- rep_col_list %>% 
          bind_cols(tibble(Gene = rownames(outputData.final.mut)), .) 
        
        data.table::fwrite(rep_col, 
                           file = paste(filename.out, "mut_training.txt", sep = "_"), 
                           sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
      }
      rm(mut.data, inputData, outputData, outputData.final.mut, rep_col_list, rep_col);gc()
      cat("Mut completed!", "\n\n")
    }
    # Methylation
    if (!is.null(meth.data)) {
        if (!is.character(meth.data[1, 1])) {
            stop("meth.data format error, please check!", call. = FALSE)
        }
        cat(c("Meth started..."), "\n")
        colnames(meth.data)[1] <- "Probe"
        ncell <- ncol(meth.data[, -1])
        if (is.null(check.cellNames)) {
            check.cellNames <- colnames(meth.data[, -1])
        } else if (length(check.cellNames) != length(colnames(meth.data[, -1])) | sum(check.cellNames %in% colnames(meth.data[, -1])) != length(check.cellNames)) {
            stop(c("Cell line names are inconsistent!"), call. = FALSE)
        }
        
        cat("Precessing", paste(length(check.cellNames), "cell lines..."), "\n")
        inputData <- meth.data[!duplicated(meth.data$Probe), ]
        
        load(paste0(path, "ccle_meth_for_missing_value_6617.RData"))
        
        outputData <- merge(meth.index, inputData, by = "Probe", sort = FALSE, all.x = TRUE)
        Probe <- outputData$Probe
        rownames(outputData) <- outputData$Probe
        value_NA <- rowSums(outputData[, -c(1, 2)])
        cat(sum(is.na(value_NA)), "genes with NA values in meth.data. Substitute by 0.",  "\n")
        if (round((sum(is.na(value_NA))/nrow(outputData)), digits = 2) > 0.2) {
            warning("NA found in >20% genes, please check if input format is correct!")
        }
        for (i in 1:nrow(outputData)) {
            if (is.na(value_NA[i])) {
              if (sum(is.na(outputData[i, ])) == sum(ncol(outputData) - 2)) {
                  outputData[i, is.na(outputData[i, ])] <- 0
                } else {
                  outputData[i, is.na(outputData[i, ])] <- 0
                }
            }
        }
        outputData <- round(as.matrix(outputData[, -1]), digits = 4)
        outputData.final.meth <- cbind(Probe, as.data.frame(outputData[, -1]))
        outputData.final.meth <- outputData.final.meth[, c("Probe", check.cellNames)]
        if (tolower(mode) == "prediction") {
            data.table::fwrite(outputData.final.meth, 
                               file = paste(filename.out, "meth_prediction.txt", sep = "_"), 
                               sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
        }
        if (tolower(mode) == "training") {
          k = 2:ncol(outputData.final.meth)
          rep_col_list <- pbmcapply::pbmclapply(X = k, FUN = function(index){
            rep_col.1 <- do.call("cbind", replicate(n, outputData.final.meth[, index], simplify = FALSE)) # 
            colnames(rep_col.1) <- paste0("C", index-1, "G", seq(1, n, 1))
            return(rep_col.1)
          }, mc.cores = 4)
          
          rep_col <- rep_col_list %>% 
            bind_cols(tibble(Probe = rownames(outputData.final.meth)), .) 
          
          data.table::fwrite(rep_col, 
                             file = paste(filename.out, "meth_training.txt", sep = "_"), 
                             sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
        }
        rm(meth.data, inputData, outputData, outputData.final.meth, rep_col_list, rep_col);gc()
        cat("Meth completed!", "\n\n")
    }
    # Gene dependency
    if (!is.character(dep.data[1, 1])) {
        stop("dep.data format error, please check!", call. = FALSE)
    } else {
        cat(c("Fingerprint started..."), "\n")
        ncell <- ncol(dep.data  [, -1])
        colnames(dep.data)[1] <- "Gene"
        idx <- which(fingerprint[1, ] %in% c("GeneSet", list.genes$Gene))
        outputData <- fingerprint[, idx]
        if (tolower(mode) == "prediction") {
            data.table::fwrite(outputData, 
                               file = paste(filename.out, "fingerprint_prediction.txt", sep = "_"), 
                               sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
        }
        if (tolower(mode) == "training") {
            
            outputData.train <- cbind(outputData[, 1], 
                                      do.call("cbind", replicate(ncell, outputData[, -1], simplify = FALSE)))
            outputData.train[1, ] <- c("GeneSet", paste0(paste0("C", rep(seq(1, ncell, 1), each = n)), "G", seq(1, n, 1)))
            data.table::fwrite(outputData.train, 
                               file = paste(filename.out, "fingerprint_training.txt", sep = "_"), 
                               sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
            
        }
        rm(outputData.train);gc()
        cat("Fingerprint completed!", "\n\n")
    }
    if (!is.null(dep.data) & tolower(mode) == "training") {
        if (is.null(check.cellNames)) {
            check.cellNames <- colnames(dep.data[, -1])
        }else if (length(check.cellNames) != length(colnames(dep.data[, -1])) | sum(check.cellNames %in% colnames(dep.data[, -1])) != length(check.cellNames)) {
            stop(c("Cell line names are inconsistent!"), call. = FALSE)
        }
        cat("Gene dependency scores (training mode) start...", "\n")
        crispr.input <- dep.data[which(dep.data$Gene %in% list.genes$Gene),  # list.genes
                                 which(colnames(dep.data) %in% c("Gene", check.cellNames))]
        k = 2:ncol(crispr.input)
        crispr.output <- pbmcapply::pbmclapply(X = k, FUN = function(index){
          table <- as.data.frame(t(crispr.input[, index]))
          colnames(table) <- paste0("C", index - 1, "G", seq(1, n, 1))
          return(table)
        }, mc.cores = 3) %>% 
          bind_cols()
          
        crispr.output <- bind_cols(tibble(Dep_Score = "score"), crispr.output)
        data.table::fwrite(crispr.output, 
                           file = paste(filename.out, "DepScore_training.txt", sep = "_"), 
                           sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
        rm(crispr.output);gc()
        cat("Gene dependency scores (training) completed!", "\n\n")
    }
    if (!is.null(cna.data)) {
        if (sum(colnames(cna.data) %in% c("CCLE_name", "Chromosome", "Start", "End", "Num_Probe", "Segment_Mean")) != 5) {
            stop("cna.data format error, please check!", call. = FALSE)
        }
        cat(c("CNA started..."), "\n")
        ncell <- length(unique(cna.data$CCLE_name))
        if (is.null(check.cellNames)) {
            outputData.cna <- .PrepCNA_custom(cna.original = cna.data, filename.out, exportTable = FALSE)
        } else {
            idx <- which(cna.data$CCLE_name %in% check.cellNames)
            if (length(check.cellNames) != length(unique(cna.data$CCLE_name[idx])) | sum(check.cellNames %in% unique(cna.data$CCLE_name[idx])) != 
                  length(check.cellNames)) {
                stop(c("Cell line names are inconsistent!"), 
                  call. = FALSE)
            }
            outputData.cna <- .PrepCNA_custom(cna.original = cna.data[idx, ], filename.out, exportTable = FALSE)
            outputData.cna <- outputData.cna[, c("CNA", check.cellNames)]
        }
        if (tolower(mode) == "prediction") {
            colnames(outputData.cna)[1] <- "Bin"
            data.table::fwrite(outputData.cna, 
                               file = paste(filename.out, "cna_prediction.txt", sep = "_"), 
                               sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
        }
        if (tolower(mode) == "training") {
          k = 2:ncol(outputData.cna)
          rep_col_list <- pbmcapply::pbmclapply(X = k, FUN = function(index){
            rep_col.1 <- do.call("cbind", replicate(n, outputData.cna[, index], simplify = FALSE)) # 
            colnames(rep_col.1) <- paste0("C", index-1, "G", seq(1, n, 1))
            return(rep_col.1)
          }, mc.cores = 4)
          
          rep_col <- rep_col_list %>% 
            bind_cols(tibble(Bin = outputData.cna$CNA), .) 
          
          data.table::fwrite(rep_col, 
                             file = paste(filename.out, "cna_training.txt", sep = "_"), 
                             sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
        }
        rm(outputData.cna, rep_col, rep_col_list, cna.data);gc()
        cat("CNA completed!", "\n\n")
    }
}

.PrepCNA_custom <- function (cna.original, filenames, exportTable = FALSE){
  cellLine <- unique(cna.original$CCLE_name)
  if (sum(tolower(colnames(cna.original)) %in% c("chr", "chromosome")) == 
      1) {
    col_idx <- which(tolower(colnames(cna.original)) %in% 
                       c("chr", "chromosome"))
    cna.original[which(tolower(cna.original[, col_idx]) == 
                         "x"), col_idx] <- 23
    cna.original[which(tolower(cna.original[, col_idx]) == 
                         "y"), col_idx] <- 24
    colnames(cna.original)[col_idx] <- "Chromosome"
  }
  chr <- unique(cna.original[, col_idx])
  chrom_length <- c(249260000, 243200000, 198030000, 191160000, 
                    180920000, 171120000, 159140000, 146370000, 141220000, 
                    135540000, 135010000, 133860000, 115170000, 107350000, 
                    102540000, 90360000, 81200000, 78080000, 59130000, 63030000, 
                    48130000, 51310000, 155280000, 59380000)
  chrom_bin <- ceiling(chrom_length/10^4)
  bigTable <- data.frame(matrix(data = 0, ncol = length(cellLine) + 
                                  1, nrow = sum(chrom_bin)), stringsAsFactors = FALSE)
  colnames(bigTable) <- c("CNA", cellLine)
  k = 1
  for (i in 1:length(chr)) {
    bin_start <- seq(0, chrom_bin[i] - 1, 1)
    bin_end <- seq(1, chrom_bin[i], 1)
    bigTable$CNA[k:(k + chrom_bin[i] - 1)] <- paste(paste0("chr", 
                                                           i), paste0(bin_start, "to", bin_end), "10k", sep = "_")
    k = chrom_bin[i] + k
  }
  t1 <- proc.time()
  path <- system.file("extdata/", package = "Prep4DeepDEP")
  load(paste0(path, "cna_table_7460.RData"))
  for (i in unique(filterTable$Chr)) {
    pb <- progress_bar$new(format = " Progress: [:bar] :percent, Estimated completion time: :eta",
                           total = ncol(bigTable), # totalnumber of ticks to complete (default 100)
                           clear = FALSE, # whether to clear the progress bar on completion (default TRUE)
                           width= 80) # width of the progress bar
    print(paste0("Chromosome : ", i))
    
    cna.filterTable <- filterTable[which(filterTable$Chr == 
                                           i), ]
    idx.chr <- which(cna.original$Chromosome == i)
    Table.chr <- cna.original[idx.chr, ]
    cna.length.chr <- (Table.chr$End - Table.chr$Start) + 
      1
    if (i == 1) {
      l = 0
    }
    else {
      l <- sum(chrom_bin[1:(i - 1)])
    }
    for (j in 2:ncol(bigTable)) {
      pb$tick()
      idx.cell <- which(Table.chr$CCLE_name == colnames(bigTable)[j])
      cellTable.chr <- Table.chr[idx.cell, ]
      cna.length <- cna.length.chr[idx.cell]
      for (k in cna.filterTable$End) {
        end_matrix <- data.frame(matrix(data = 1, nrow = nrow(cellTable.chr)) * 
                                   10^4 * k, stringsAsFactors = FALSE)
        end_matrix$cellTable <- cellTable.chr$End
        start_matrix <- data.frame(matrix(data = 1, nrow = nrow(cellTable.chr)) * 
                                     10^4 * (k - 1) + 1, stringsAsFactors = FALSE)
        start_matrix$cellTable <- cellTable.chr$Start
        overlap.length <- (10^4) + cna.length - (apply(end_matrix, 
                                                       1, max) - apply(start_matrix, 1, min) + 1)
        overlap.length[overlap.length < 0] <- 0
        bigTable[l + k, j] <- round(sum(overlap.length * 
                                          cellTable.chr$Segment_Mean)/10^4, digits = 4)
      }
    }
  }
  bigTable.filter <- merge(filterTable, bigTable, by = "CNA", 
                           all.x = TRUE, sort = FALSE)
  bigTable <- bigTable.filter[, -c(2:4)]
  if (exportTable == TRUE) {
    write.table(bigTable, file = paste(filenames, nrow(bigTable.filter), 
                                       "_CNA_filter.txt", sep = "_"), sep = "\t", col.names = TRUE, 
                row.names = FALSE, quote = FALSE)
  }
  t2 <- proc.time()
  t <- round((t2 - t1)/60, digits = 2)
  print(c("Computation time (mins)", t[1:3]))
  return(bigTable)
}

depoi_extraction <- function(GD_PATH = "/home/wmbio/WORK/gitworking/Dependency_prediction/prediction/data/paper/gene_effect.csv",
                             MUT_PATH = "/home/wmbio/WORK/gitworking/Dependency_prediction/prediction/data/paper/ccl_mut_data_paired_with_tcga.txt",
                             DepOI_SD = 0.15, remove_default_OI = FALSE){
  
  # mutation
  ccls <- data.table::fread(file = MUT_PATH, sep = "\t") %>% colnames() %>% .[-1]
  
  # gene dependency
  gd <- data.table::fread(file = GD_PATH, sep = ",") %>%  
    separate(cell_line_name, sep = "_", into = c("cell_line_name", "B","C"), ) %>% 
    select(-B, -C) %>% 
    filter(cell_line_name %in% ccls)
  
  gd_genes <- gd %>% colnames() %>% 
    lapply(X = ., FUN = function(value){
      str_split(string = value, pattern = " \\(") %>% 
        unlist() %>% .[1] %>% return()
    }) %>% unlist()
  
  gd_matrix <- gd %>% select(-1) %>% t()
  rownames(gd_matrix) <- gd_genes[-1]
  
  dep.data_custom <- apply(gd_matrix, MARGIN = 1, sd, na.rm = TRUE) %>% 
    tibble(Gene = rownames(gd_matrix), SD = .) %>% 
    filter(SD >= DepOI_SD) %>% 
    select(Gene)
  
  if(remove_default_OI){
    path <- system.file("extdata/", package = "Prep4DeepDEP")
    load(paste0(path, "default_dep_genes_1298.RData"))
    dep.data_custom <- setdiff(dep.data_custom[[1]], dep.data[[1]]) %>% tibble(Gene = .)
  }
  
  return(dep.data_custom)
}
