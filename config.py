import os
import numpy
import torch
"""
configuration file includes all related datasets 
"""
CUDA_ID = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#root_data_folder = 'data/'
#raw_data_folder = os.path.join(root_data_folder, 'raw_dat')
#preprocessed_data_folder = os.path.join(root_data_folder, 'preprocessed')
#gene_feature_file = os.path.join(preprocessed_data_folder, 'CosmicHGNC_list.tsv')

#TCGA_datasets
gene_feature_file = 'CosmicHGNC_list.tsv'
tcga_multi_label_file = 'TCGA_labels.csv'
tcga_sample_file = 'TCGA_phenotype_denseDataOnlyDownload.tsv.gz'

#CCLE datasets
ccle_gex_file = 'CCLE/zscore.csv'
ccle_preprocessed_gex_file = 'CCLE_expression.csv'
ccle_sample_file = 'sample_info.csv'

#gex features
gex_feature_file = '1000gene_features.csv'

#GDSC datasets
gdsc_target_file1 = 'GDSC1_fitted_dose_response_25Feb20.csv'
gdsc_target_file2 = 'GDSC2_fitted_dose_response_25Feb20.csv'
gdsc_raw_target_file = 'gdsc_ic50flag.csv'
gdsc_sample_file = 'gdsc_cell_line_annotation.csv'
gdsc_preprocessed_target_file = 'gdsc_ic50flag.csv'
gdsc_drugs = 'drug_smiles.csv'

label_graph = numpy.array(object=object)
label_graph_norm = numpy.array(object=object)

tissue_map = numpy.array(object=object)
drug_feat = numpy.array(object=object)
