
# import modules
import os
import glob
from tqdm import tqdm
import re

import numpy as np
import pandas as pd
import sys
sys.path.append('/home/samuelA/.local/lib/python3.10/site-packages')

import nibabel as nib

import os
import subprocess


def make_fsaverage(input_path, output_path, file_type, ico=7):
    '''
    given some input path and output path, convert the inputted file to native space (ico-7)

    param input_path: the working directory, where the files you want modified are (should have: mri, label, surf, etc.)
    param output_path: where to output the converted file or where the converted file is
    param ico: the target icosahedron order

    return: path to converted file if successful, None otherwise
    '''

    subj_dir = -3  # last directory before the surf folder
    freesurfer_dir = '/'.join(input_path.split('/')[:subj_dir])
    set_dir = f"export SUBJECTS_DIR='{freesurfer_dir}'; "
    surf = 'mri_surf2surf'

    subject = ' --s ' + (input_path.split('/')[subj_dir]) + ' '
    hemi = ' --hemi ' + input_path.split('/')[-1][:2]
    srcsurfval = f' --srcsurfval {input_path} '

    if 'w-g.pct.mgh' in input_path:
        output_fname_abs = f'{output_path}{input_path.split("/")[subj_dir-2]}_{input_path.split("/")[subj_dir]}_{hemi[-2:]}.w-g.pct.mgh'
    else:
        output_fname_abs = f'{output_path}{input_path.split("/")[subj_dir-2]}_{input_path.split("/")[subj_dir]}_{hemi[-2:]}_{".".join(input_path.split(".")[-1:])}.mgh'

    target_subj = '--trgsubject ico '
    trig_order = f'--trgicoorder {ico} '
    log_output = f' > {output_path}last_conversion.log'

    # IXI specific to filter ID

    id_start = output_fname_abs.find("all/IXI")+7
    underscore_pos = output_fname_abs.find("_", id_start)
    output_fname_abs = f'{output_fname_abs[:id_start-3]}{output_fname_abs[id_start:id_start+3]}{output_fname_abs[underscore_pos:]}'
    ###

    # If file corruption
    if not validate_inputs(input_path, file_type[:-3]):
        if os.path.exists(output_fname_abs):
            os.remove(output_fname_abs)
        return None

    output_fname_cmd = ' --o ' + output_fname_abs
    shell_command = set_dir + surf + subject + target_subj + trig_order + hemi + srcsurfval + output_fname_cmd + log_output
    result = subprocess.run(shell_command, shell=True, executable='/bin/bash')
        
    if result.returncode == 0:
        if os.path.isfile(output_fname_abs):
            return output_fname_abs
        else:
            return None
    else:
        if os.path.exists(output_fname_abs):
            os.remove(output_fname_abs)
        return None
            
    if os.path.exists(output_fname_abs):
        os.remove(output_fname_abs)
    return None

def validate_inputs(input_path, file_type):

    file_type_ranges = {
        'area': {'min': 0, 'max': 100},
        'curv': {'min': -40, 'max': 40},
        'w-g.pct.mgh': {'min': -200, 'max': 200},
        'sulc': {'min': -20, 'max': 20},
        'thickness': {'min': 0, 'max': 10}
    }
    
    # Verify that the file values are within the expected range
    min_val = file_type_ranges[file_type]['min']
    max_val = file_type_ranges[file_type]['max']
    
    try:
        # For MGH/MGZ files
        if file_type.endswith('.mgh') or file_type == 'w-g.pct.mgh':
            img = nib.load(input_path)
            data = img.get_fdata()

        # For FreeSurfer curvature files
        elif file_type in ['curv', 'area', 'sulc', 'thickness']:
            data = nib.freesurfer.read_morph_data(input_path)
        
        # For surface geometry files
        elif file_type in ['white', 'pial', 'inflated']:
            verts, faces = nib.freesurfer.read_geometry(input_path)
            data = verts  # Validate vertex coordinates

        else:
            print(f"Unsupported file type: {file_type}")
    
    except KeyError:
        return False

    # Verify within expected range
    if np.any(data < min_val) or np.any(data > max_val):
        print(f"Value out of range in {input_path}")
        print(f"min_val: {np.min(data)}\n max_val: {np.max(data)}")
        return False
    
    return True

def make_data(root_path='/mnt/md0/tempFolder/samAnderson/', file_types=['w-g.pct.mgh', 'curv', 'thickness'], # used to be ratio so older sorts are incorrect
             data_path='lab/lab_organized/subject-*/freesurfer_output/*/',
             output_dir=None,
             damaged_subjects = ['None'], ico=7):
    
    '''
    function for projecting surf files to ico-7
    returns dictionary
    '''

    # alphabetize file types for consistency across datasets
    file_types = sorted(file_types)

    # create a dictionary with the subject paths
    all_paths = {}

    for f in file_types:

        rh_temp = glob.glob(root_path + data_path + f'surf/rh.{f}') # right hemisphere
        lh_temp = glob.glob(root_path + data_path + f'surf/lh.{f}') # left hemisphere

        # remove damaged subjects (where not all files are available)
        for subj in damaged_subjects:
            rh_temp = [x for x in rh_temp if subj not in x]
            lh_temp = [x for x in lh_temp if subj not in x]

        all_paths[f'{f}_rh'] = sorted(rh_temp) # so that all of the paths align
        all_paths[f'{f}_lh'] = sorted(lh_temp) # rh_curv, lh_curv, rh_w-g, lh_w-g, rh_thickness, lh_thickness

    # set the output directory, then either make or load in atlas-projected subjects
    if output_dir == None:
        output_dir = root_path + f'gnn_model/datasets/{data_path.split("/")[0]}/'  

    # create the output directory
    try: os.mkdir(output_dir)
    except FileExistsError: pass

    # get the data from each of the paths, or convert the data and get the location data
    damaged_subjects = []
    for file_type in tqdm(all_paths, desc="Converting file types.."): # rh.curv, lh.curv, etc.
        for file in all_paths[file_type]: # each individual subject/timepoint combination
            damaged_subjects.append(make_fsaverage(file, output_dir, file_type, ico))

    return [item for item in damaged_subjects if item is not None] # return the damaged subjects

def make_npy(dataset_dir, 
              metadata_path, 
              file_suffixes = ['_lh_curv.mgh', '_rh_curv.mgh', '_lh_thickness.mgh', '_rh_thickness.mgh', '_lh.w-g.pct.mgh', '_rh.w-g.pct.mgh'],
              age_id_date=[],
              massive=False,
              chunk_path=None,
              first_hemi='rh'):
    '''
    This function is intended to load the processed files
    get_data does technically work aswell but it can overload the system, and takes a while

    param dataset_dir: the path containing the processed files, str
    param metadata_path: the path to the metadata, must have subject id and age
    param file_suffixes: a list of
    all of the file types we want each valid subject to have
    param age_id_date: the column titles for the age, id, and date
    param massive: whether to split the data into 100ths, and save the resulting arrays as pickle files 
    param chunk_path: where to output chunk files
    param first_hemi: which hemi should be first in the resulting array, i.e. is node 0 right or left?

    return: X, y
    '''

    # find which subjects were not completely processed

    subject_files = {}
    
    for suffix in file_suffixes:
        files = glob.glob(os.path.join(dataset_dir, f'*{suffix}'))
        subject_ids = {os.path.basename(f).replace(suffix, '') for f in files}
        subject_files[suffix] = subject_ids

    # find all unique subject IDs
    all_subject_ids = set()
    for ids in subject_files.values():
        all_subject_ids.update(ids)

    # find subject IDs that are missing from the complete set
    subjects_without_all_files = [subject for subject in all_subject_ids if not all(subject in subject_files[suffix] for suffix in file_suffixes)]
    #print(f"Subjects without all files: {subjects_without_all_files}")

    # alphabetize the file suffixes ignoring hemisphere for consistency
    file_suffixes = sorted(file_suffixes, key=lambda x: x[4:])

    # load in the demographic info / metadata

    if '.xlsx' in metadata_path:
        metadata_csv = pd.read_excel(metadata_path, dtype=str)
    elif 'csv' in metadata_path:
        metadata_csv = pd.read_csv(metadata_path, dtype=str)
    else:
        raise Exception('Error: Invalid metadata file type')
    age_column, id_column, date_column = age_id_date

    ##### MODIFY AS NEEDED #####

    # remove the .0 from the dates
    #metadata_csv[date_column] = metadata_csv[date_column].astype(str).str.replace(r"\.0$", "", regex=True)

    # remove non-CNs
    #subjects_without_all_files.extend(metadata_csv[metadata_csv['TBI_status'] != 'Control'].apply(lambda row: f"{row[id_column]}_{row[date_column]}", axis=1).tolist())

    # convert x/y/zz to ZZZZYYXX assuming it refers to 20XX for the date
    metadata_csv[date_column] = metadata_csv[date_column].apply(
    lambda x: (
        f"20{x.split('/')[2].zfill(2)}"
        f"{x.split('/')[0].zfill(2)}"
        f"{x.split('/')[1].zfill(2)}"
    ) if pd.notna(x) else None )

    ###########################

    # get subjects
    if dataset_dir[-1] != '/': dataset_dir.append('/')
    all_files = glob.glob(dataset_dir + f'*{file_suffixes[0]}') # [0] is arbitrary, to get subjects, make sure all
    subject_ids = [os.path.basename(f).replace(file_suffixes[0], '') for f in all_files]

    # remove damaged subjects
    try: subject_ids = [s for s in subject_ids if s not in subjects_without_all_files]
    except ValueError: print('Value Error')

    # get the number of nodes
    sample_file = nib.load(all_files[0]).get_fdata()
    num_nodes = sample_file.shape[0]

    if massive:
    
    	# create the output directory
        try: os.mkdir(chunk_path)
        except FileExistsError: pass
        
        chunk_size = 500
        chunked_list = [subject_ids[i:i + chunk_size] for i in range(0, len(subject_ids), chunk_size)]

        for chunk_idx, chunk in enumerate(chunked_list):
            retained_subject_ids = chunk.copy()
            X, y = get_X_and_y(num_nodes, chunk, date_column, 
                               metadata_csv, age_column, id_column, 
                               retained_subject_ids, file_suffixes, dataset_dir, first_hemi)
            
            # normalize X
            add_norm_across_all_files()

            # save the chunks as files
            np.save(f'{chunk_path}/X_NACC_AD_{chunk_idx}.npy', X)
            np.save(f'{chunk_path}/y_NACC_AD_{chunk_idx}.npy', y)
        return X, y

    else:
        retained_subject_ids = subject_ids.copy()
        X, y, damaged_subjects = get_X_and_y(num_nodes, subject_ids, date_column, 
                           metadata_csv, age_column, id_column, 
                           retained_subject_ids, file_suffixes, dataset_dir, first_hemi)
        return X, y, damaged_subjects

def get_X_and_y(num_nodes, subject_ids, date_column, 
                           metadata_csv, age_column, id_column, 
                           retained_subject_ids, file_suffixes, dataset_dir,
                           first_hemi):
        
    num_subjects = len(subject_ids)
    num_features = len(file_suffixes)

    # initialize the main data array
    X = np.zeros((num_subjects, (num_nodes*2), (num_features//2))) # to account for the hemispheres being independent nodes

    # initialize the empty age list
    y = []

    # save the damaged subjects (that have nan in them)
    damaged_subjects = []

    # for each subject
    for subj_idx, subject_id in enumerate(subject_ids):
        
        ####
        # Update y
        ####

        # add the age to the corresponding list
        if date_column == None: age_df = metadata_csv[[age_column, id_column]]
        else: age_df = metadata_csv[[age_column, id_column, date_column]]

        # get the working file's date and subject id
        target_id, target_date = subject_id.rsplit('_', 1)

        # get the desired row
        if date_column == None: age_value = age_df[(age_df[id_column] == target_id)]
        else: age_value = age_df[(age_df[id_column] == target_id) & (age_df[date_column] == target_date)]
        try:
            if 'Y' in age_value[age_column].values[0]: # from '64Y' format if relevant
                as_float = float(age_value[age_column].values[0][1:-1])
            else:
                as_float = float(age_value[age_column].values[0])

            # add the age value
            y.append(as_float) # from '64Y' format
            retained_subject_ids.append(subject_id)
        except IndexError: # if there isn't an associated age value, we don't consider this subject
            continue

        ####
        # Update X
        ####

        # for each file type
        for suffix_idx, suffix in enumerate(file_suffixes):

            # get the path of the target file
            file_path = os.path.join(dataset_dir, f'{subject_id}{suffix}')

            # load in the data
            data = nib.load(file_path).get_fdata().flatten() # flatten: from (163842, 1, 1) to (163842)
            # if referencing the left hemisphere, i.e. the earlier nodes

            if first_hemi in suffix:
                X[subj_idx, :(X.shape[1]//2), suffix_idx//2] = data
            else:
                X[subj_idx, (X.shape[1]//2):, suffix_idx//2] = data

            # check if damaged
            if np.isnan(data).any():
                damaged_subjects.append(subject_id)

    # make y an numpy array
    y = np.array(y)

    # get rid of the excess 0s due to skipped subjects
    non_zero_subjects = np.any(X != 0, axis=(1, 2))
    X = X[non_zero_subjects]

    return X, y, damaged_subjects # array, array


### run the preprocessing pipeline ###

# ico level to start at
ico=6
# file types
file_types=['area', 'curv', 'w-g.pct.mgh', 'sulc', 'thickness']
hemispheres = ['_lh', '_rh']
# to account for w-g being named differently
file_suffixes = [
    f"{hemi}.{feature}" if feature == "w-g.pct.mgh" else f"{hemi}_{feature}.mgh"
    for feature in file_types
    for hemi in hemispheres]


###

folder = f'IXI_ico{ico}_all'
raw_data_dir = f'/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/datasets/raw/{folder}/'
"""
damaged_subjects = make_data(root_path='/mnt/md0/subjectdata/', file_types=file_types, # used to be ratio so older sorts are incorrect
             data_path='IXI/*/freesurfer_output/*/',
             output_dir=raw_data_dir,
             damaged_subjects = ['None'], ico=ico)
print(f'Number of damaged subjects: {len(damaged_subjects)}')
"""

X, y, damaged_subjects = make_npy(dataset_dir=raw_data_dir, 
              metadata_path='/mnt/md0/tempFolder/samAnderson/datasets/IXI_master.csv',
              file_suffixes = file_suffixes,
              age_id_date=['AGE', 'IXI_ID', 'STUDY_DATE'],
              massive=False,
              first_hemi='rh')

###

# validate X and y
print(X.shape)
print(y.shape)

# X has shape (subj, vertices, features)
# Calculate mean and std separately for each feature (last dimension)
mean_X = np.mean(X, axis=(0, 1), keepdims=True)  # shape (1, 1, features)
std_X = np.std(X, axis=(0, 1), keepdims=True)    # shape (1, 1, features)

# Normalize X seperately by feature (they vary vastly)
X_norm = (X - mean_X) / std_X

np.save(f'/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/datasets/processed/X_IXI', X_norm)
np.save(f'/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/datasets/processed/y_IXI', y)