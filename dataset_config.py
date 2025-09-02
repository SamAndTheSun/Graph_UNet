import copy

UKBB = {
    'name': 'UKBB',
    'set': 'training',
    'raw_data': '/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/datasets/raw/UKBB_ico6_all_pruned/',
    'metadata': '/mnt/md0/tempFolder/samAnderson/datasets/UKBB_demographic_with_sex.csv',
    'id_col': 'eid',
    'date_col': 'date',
    'age_col': 'age',
    'sex_col': 'sex',
    'sex_mapping': {'Female': 0, 'Male': 1},
    'data_preproc': ['all_str'],
    'select': 'all'
}

IXI = {
    'name': 'IXI',
    'set': 'training',
    'raw_data': '/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/datasets/raw/IXI_ico6_all/',
    'metadata': '/mnt/md0/tempFolder/samAnderson/datasets/IXI_master.csv',
    'id_col': 'IXI_ID',
    'date_col': 'STUDY_DATE',
    'age_col': 'AGE',
    'sex_col': 'SEX_ID (1=m, 2=f)',
    'sex_mapping': {'Female': 2, 'Male': 1},
    'data_preproc': ['DD/MM/YY conversion'],
    'select': 'all'
}

NACC = {
    'name': 'NACC',
    'set': 'training',
    'raw_data': '/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/datasets/raw/NACC_ico6_all/',
    'metadata': '/mnt/md0/tempFolder/samAnderson/datasets/NACC_master.csv',
    'id_col': 'ID',
    'date_col': 'study_time',
    'age_col': 'age',
    'sex_col': 'sex',
    'sex_mapping': {'Female': 'Female', 'Male': 'Male'},
    'data_preproc': ['remove_.0'],
    'select': 'TBI_status==Control'
}

SLIM = {
    'name': 'SLIM',
    'set': 'pretraining',
    'raw_data': '/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/datasets/raw/SLIM_ico6_all/',
    'metadata': '/mnt/md0/tempFolder/samAnderson/datasets/SLIM_master.csv',
    'id_col': 'ID',
    'date_col': 'Date',
    'age_col': 'Age',
    'sex_col': 'Sex',
    'sex_mapping': {'Female': 'female', 'Male': 'male'},
    'data_preproc': None,
    'select': 'all'
}

ID1000 = {
    'name': 'ID1000',
    'set': 'pretraining',
    'raw_data': '/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/datasets/raw/AOMIC_ID1000_ico6_all/',
    'metadata': '/mnt/md0/tempFolder/samAnderson/datasets/aomic_id1000_master.csv',
    'id_col': 'participant_id',
    'date_col': None,
    'age_col': 'age',
    'sex_col': 'sex',
    'sex_mapping': {'Female': 'female', 'Male': 'male'},
    'data_preproc': None,
    'select': 'all'
}

ADNI_CN = {
    'name': 'ADNI_CN',
    'set': 'testing',
    'raw_data': '/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/datasets/raw/ADNI_ico6_all/',
    'metadata': '/mnt/md0/tempFolder/samAnderson/datasets/ADNI1-4/ADNI1-4_master.csv',
    'id_col': 'Subject ID',
    'date_col': 'Study Date',
    'age_col': 'Age',
    'sex_col': 'Sex',
    'sex_mapping': {'Female': 'F', 'Male': 'M'},
    'data_preproc': ['full_date'],
    'select': 'Research Group==CN'
}

ADNI_AD = copy.deepcopy(ADNI_CN)
ADNI_AD['name'] = 'ADNI_AD'
ADNI_AD['select'] = 'Research Group==AD'

# The final list you want to load directly
datasets = [UKBB, NACC, IXI, SLIM, ID1000, ADNI_CN, ADNI_AD]
