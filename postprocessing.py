# Standard library imports
import os
import sys
import glob
import subprocess

# Plotting imports
import seaborn as sns
import matplotlib.pyplot as plt

# Custom paths
sys.path.append('/home/samuelA/.local/lib/python3.10/site-packages')
sys.path.append('/mnt/md0/tempFolder/samAnderson/gnn_model')

# Third-party imports
import numpy as np
import pandas as pd
import nibabel as nib

# Statistical third-party imports
import scipy.io
from scipy.stats import skew
import scipy.sparse as sparse

# PYG imports
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Function for saving per_node_mae as MATLAB files for post-processing, averaged by label
def get_matlab(per_node_values, output_prefix=None, first='rh'):
    '''
    Save per-node data as MATLAB files, per ico
    '''
                
    # Put left hemisphere first (assumed in nahian's code)
    if first == 'rh':
        assert len(per_node_values) % 2 == 0 # verify that there are an even number of vertices
        half = len(per_node_values) // 2 # find the halfway point
        right_hemisphere = per_node_values[:half] # select the first half
        left_hemisphere = per_node_values[half:] # select the second half
        # Concat with left hemisphere first
        ico_vertices = np.concatenate((left_hemisphere, right_hemisphere)) # swap the halves
    else: # if lh is already first
        pass
        
    # Save to a .mat file
    mat_filename = f"{output_prefix}.mat"
    scipy.io.savemat(mat_filename, {'data': ico_vertices})
        
    #print(f"Saved {mat_filename}")
    return

# Function for masking outliers based on distribution
def clip_outliers(arr, min_percentile=1, max_percentile=99):
    lower_bound = np.percentile(arr, min_percentile)
    upper_bound = np.percentile(arr, max_percentile)
    return np.clip(arr, lower_bound, upper_bound)

# Function for getting the average error, variance, and skew per region
def get_region_stats(per_node_e, per_node_e_corrected=None, pred_per_vertex=None, set='5cv', ico=6, first='rh', abs=True):

    # Get label data
    if ico == 7:
        fsavg_path = '/mnt/md0/softwares/freesurfer/subjects/fsaverage/'
    else:
        fsavg_path = f'/mnt/md0/softwares/freesurfer/subjects/fsaverage{ico}/'
    
    rh_labels, _, rh_names = nib.freesurfer.read_annot(f'{fsavg_path}label/rh.aparc.a2009s.annot')
    lh_labels, _, lh_names = nib.freesurfer.read_annot(f'{fsavg_path}label/lh.aparc.a2009s.annot')

    # Combine hemispheres with tracking
    if first == 'rh':
        labels = np.hstack((rh_labels, lh_labels + rh_labels.max() + 1))
        names = [(n.decode('utf-8'), 'rh') for n in rh_names] + [(n.decode('utf-8'), 'lh') for n in lh_names]
    else:
        labels = np.hstack((lh_labels, rh_labels + lh_labels.max() + 1))
        names = [(n.decode('utf-8'), 'lh') for n in lh_names] + [(n.decode('utf-8'), 'rh') for n in rh_names]

    unique_labels = np.unique(labels)
    rows = []

    if pred_per_vertex is None:
        for lid in unique_labels:

            # Mask for the target region
            region_mask = (labels == lid)
            region_name, hemi = names[lid]
            
            # Get the average error, variance, and skew of the region predictions
            error = per_node_e[region_mask].mean()

            rows.append({
                "set": set,
                "freesurfer region": region_name,
                "hemi": hemi,
                "age gap": f'{error:.2f}',
                "variance": "-",
                "skew": "-",
                "sort_val" : f'{error:.2f}'
            })

    else:
        for lid in unique_labels:

            # Mask for the target region
            region_mask = (labels == lid)
            region_name, hemi = names[lid]

            # Get the average error, variance, and skew of the region predictions
            error = per_node_e[region_mask].mean()
            corrected_error = per_node_e_corrected[region_mask].mean()
            var = np.var(pred_per_vertex[:, region_mask], axis=0, ddof=1).mean()
            skew_val = skew(pred_per_vertex[:, region_mask], axis=0).mean()

            rows.append({
                "set": set,
                "freesurfer region": region_name,
                "hemi": hemi,
                "age gap": f'{corrected_error:.2f} ({error:.2f})',
                "variance": f'{var:.2f}',
                "skew": f'{skew_val:.2f}',
                "sort_val" : corrected_error
            })

    # Create the final DataFrame
    if abs: df = pd.DataFrame(rows).sort_values(by="sort_val", key=lambda x: x.abs(), ascending=False).drop(columns=['sort_val'])
    else: df = pd.DataFrame(rows).sort_values(by="sort_val", ascending=False).drop(columns=['sort_val'])

    return df

# Function for correcting bias based on difference between predictions and actual (semi-local bias correction)
def bias_correction(chr_ages, pred_per_vertex, factors=None, method='behesti'):

    # Input validation
    assert len(chr_ages) == pred_per_vertex.shape[0], "Mismatch in number of subjects"
    chr_ages_reshaped = chr_ages[:, np.newaxis]  # For broadcasting

    # Design matrix for linear regression (n_subjects, 2)
    X = np.column_stack([chr_ages, np.ones_like(chr_ages)])
    
    if method == 'behesti': # Behesti et al., 2019

        if factors is None:
            # Vectorized computation of age gap (n_subjects, n_vertices)
            age_gap = pred_per_vertex - chr_ages_reshaped
            
            # Vectorized least-squares solve for all vertices (2, n_vertices)
            coefficients, _, _, _ = np.linalg.lstsq(X, age_gap) # add an extra _ if using a newer np version

            # Get the average slope and bias
            avg_m = np.mean(coefficients[0])
            avg_b = np.mean(coefficients[1])
            factors = np.array([avg_m, avg_b])

            # Print out the factors
            print(f'Factors: {factors}')
        
        # Apply global correction
        all_corrected = pred_per_vertex - ((factors[0] * chr_ages_reshaped) + factors[1])
        
    elif method == 'cole': # Cole et al., 2018
        
        if factors is None:

            # Vectorized solve for all vertices
            coefficients = np.linalg.lstsq(X, pred_per_vertex)[0]
            
            # Average local slopes (m) and intercepts (b)
            avg_m = np.mean(coefficients[0])
            avg_b = np.mean(coefficients[1])
            factors = np.array([avg_m, avg_b])  # Global (2,) factors

            # Print out the factors
            print(f'Factors: {factors}')
            
        # Apply per-vertex correction: (pred - b)/m
        all_corrected = (pred_per_vertex - factors[1]) / factors[0]
    
    # Compute errors
    corrected_e = np.mean(all_corrected - chr_ages_reshaped, axis=0)
    corrected_age_gap = np.mean(all_corrected, axis=1) - chr_ages
    
    return corrected_e, corrected_age_gap, factors

# Function for plotting global age gaps
def age_gap_plot(age_gaps, output_dir, suffix, min_x=-20, max_x=20):

    # Set style and limits
    sns.set_style("white")
    plt.xlim(min_x, max_x)
    
    # Set font sizes
    title_fontsize = 14
    label_fontsize = 14
    
    # Show the distribution of age gaps [Global: BA-CA] with KDE only
    sns.kdeplot(age_gaps, color='blue', alpha=0.5, linewidth=2, fill=True)

    # Plot labelling with larger fonts
    """
    if 'corrected' in output_dir:
        plt.title("Corrected Global Age Gap (BA' - CA)", fontsize=title_fontsize)
    else:
        plt.title("Global Age Gap (BA - CA)", fontsize=title_fontsize)
    """
    plt.xlabel("Age Gap", fontsize=label_fontsize)
    plt.ylabel("Density", fontsize=label_fontsize)

    # Format statistics, save plot, clear figure, and return
    plt.savefig(f"{output_dir}{suffix}_age_gaps.png", dpi=300, bbox_inches='tight')
    print(f'Saved Figure: {output_dir}{suffix}_age_gaps.png')
    print(f'Figure stats: mean = {np.mean(age_gaps)} ; median = {np.median(age_gaps)} ; std = {np.std(age_gaps)} ; var = {np.var(age_gaps)}')
    plt.clf()

    return

# Function for smoothing out prediction values
def smooth_vertex_data(pred_per_vertex, chr_ages, n_iter=4, hops=2, ico=6):

    # 1. Smooth the mesh values
    fsavg_path = f'/mnt/md0/softwares/freesurfer/subjects/fsaverage{ico}/'
    _, faces = nib.freesurfer.read_geometry(f'{fsavg_path}surf/rh.pial')
    
    # Stack the faces to account for both hemispheres (same for each hemi)
    faces = np.vstack((faces, faces + (np.max(faces) + 1)))

    # Convert to 0-based if needed
    if faces.min() == 1:
        faces = faces - 1
    n_verts = pred_per_vertex.shape[1]

    # Build adjacency matrix (only once)
    rows = np.hstack([faces[:,0], faces[:,1], faces[:,2]])
    cols = np.hstack([faces[:,1], faces[:,2], faces[:,0]])
    adj = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n_verts, n_verts)
    )
    adj = adj.maximum(adj.T)  # Ensure symmetry
    
    # Create multi-hop neighborhood matrix
    neighborhood = sparse.eye(n_verts)  # Start with self-connections
    if hops > 0:
        hop_matrix = adj.copy()
        for _ in range(hops-1):
            hop_matrix = hop_matrix @ adj
        neighborhood += hop_matrix
        
    # Normalize by degree (including self)
    degrees = np.array(neighborhood.sum(axis=1)).ravel()
    inv_degrees = sparse.diags(1.0 / degrees)
    
    # Create smoothing operator
    smoothing_op = inv_degrees @ neighborhood
    
    # Apply smoothing to each subject
    smoothed_pred = pred_per_vertex.copy()
    for _ in range(n_iter):
        smoothed_pred = smoothed_pred @ smoothing_op.T
    
    # 2. Compute age gaps (mean prediction vs chronological age)
    vertex_means = np.mean(smoothed_pred, axis=1)
    age_gaps = vertex_means - chr_ages
    
    # 3. Compute per-node errors (mean prediction error per vertex)
    per_node_e = np.mean(smoothed_pred - chr_ages[:, np.newaxis], axis=0)
    
    return smoothed_pred, age_gaps, per_node_e

# Function for post-processing
def postprocess(chr_ages, age_gaps, pred_per_vertex, output_dir, suffix, factors=None, abs=True, ico=6):

    # Smooth the predictions
    pred_per_vertex, age_gaps, smoothed_e = smooth_vertex_data(pred_per_vertex, chr_ages, ico=6)

    # Show the distribution of age gaps [Global: BA-CA]
    age_gap_plot(age_gaps, output_dir, f'{suffix}_raw')

    # Clip the errors for visualization purposes
    clipped_smoothed_e = clip_outliers(smoothed_e)

    # Save the smoothed, clipped error as a matlab array
    get_matlab(clipped_smoothed_e, output_prefix=f'{output_dir}{suffix}_raw_ME_data')

    # Run bias correction
    corrected_e, corrected_age_gap, factors = bias_correction(chr_ages, pred_per_vertex, factors)

    # Show the distribution of age gaps [Global: BA-CA]
    age_gap_plot(corrected_age_gap, output_dir, f'{suffix}_corrected')

    # Clip the errors for visualization purposes
    clipped_corrected_e = clip_outliers(corrected_e)

    # Save the corrected error as a matlab array
    get_matlab(clipped_corrected_e, output_prefix=f'{output_dir}{suffix}_corrected_ME_data')

    # Convert to MATLAB cell array syntax
    mat_files = [f'{output_dir}{suffix}_raw_ME_data', f'{output_dir}{suffix}_corrected_ME_data']
    matlab_file_list = "{" + ",".join([f"'{f}'" for f in mat_files]) + "}"

    # Run the MATLAB code
    command = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'lat_L','lat_R','med_R','med_L'}}); exit"]
    result = subprocess.run(command, cwd="/mnt/md0/tempFolder/samAnderson/nahian_code/", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    #print(result)

    paths = [
        f'{output_dir}{suffix}_raw_ME_data_latL_latR_medR_medL.png', 
        f'{output_dir}{suffix}_raw_age_gaps.png',
        f'{output_dir}{suffix}_corrected_ME_data_latL_latR_medR_medL.png',
        f'{output_dir}{suffix}_corrected_age_gaps.png'
    ]

    # Save the alternative angles
    command = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'ant','dor','pos','ven'}}); exit"]
    result = subprocess.run(command, cwd="/mnt/md0/tempFolder/samAnderson/nahian_code/", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    #print(result)
    
    # Get the stats by region, retaining smoothing but ignoring clipping
    region_stats_df = get_region_stats(smoothed_e, corrected_e, pred_per_vertex, set=suffix, abs=abs)

    # Return the paths of the images to plot as a list, and the regional df
    return paths, region_stats_df, factors

# Function for getting dataset statistics based on what files were converted
def get_dataset_statistics(datasets, age_filter=None):
    
    # Initialize the summary DataFrame with correct columns
    d_table = pd.DataFrame(columns=['repository', 'set', 'N', 'min', 'max', 'μ', 'σ', 'M:F'])

    # Store the training demographic data for depicting the combined set
    training_data = []

    # Define the raw file types
    file_types = ['area', 'curv', 'w-g.pct.mgh', 'sulc', 'thickness']
    hemispheres = ['_lh', '_rh']
    file_suffixes = [
        f"{hemi}.{feature}" if feature == "w-g.pct.mgh" else f"{hemi}_{feature}.mgh"
        for feature in file_types
        for hemi in hemispheres
    ]

    # Begin counting number of males and females
    # we need this because some have overlapping mappings (i.e. 0 is female sometimes male others)
    training_sex_counts = {'Female' : 0, 'Male' : 0}
    
    for dset in datasets:
        # Get the metadata
        if dset['metadata'][-4:] == '.csv':
            df = pd.read_csv(dset['metadata'])
        elif dset['metadata'][-5:] == '.xlsx':
            df = pd.read_excel(dset['metadata'])
        
        # Apply preprocessing if specified
        if dset['data_preproc'] is not None:

            if 'DD/MM/YY conversion' in dset['data_preproc']:
                df[dset['date_col']] = df[dset['date_col']].apply(
                    lambda x: (
                        f"20{x.split('/')[2].zfill(2)}"
                        f"{x.split('/')[0].zfill(2)}"
                        f"{x.split('/')[1].zfill(2)}"
                    ) if pd.notna(x) else None
                )

            elif 'remove_.0' in dset['data_preproc']:
                df[dset['date_col']] = df[dset['date_col']].astype(str).str.replace(r"\.0$", "", regex=True)

            elif 'full_date' in dset['data_preproc']:
                df[dset['date_col']] = pd.to_datetime(df[dset['date_col']], dayfirst=True).dt.strftime('%Y%m%d').astype(int)

            elif 'all_str': # UKBB is weird and needs everything except age as a str beforehand
                df[dset['date_col']] = df[dset['date_col']].astype(str)
                df[dset['sex_col']] = df[dset['sex_col']].astype(str)
                df[dset['id_col']] = df[dset['id_col']].astype(str)

        # Find raw files and determine split position
        raw_files = glob.glob(f'{dset["raw_data"]}*')
        parts = raw_files[0][len(dset["raw_data"]):].split('_')
        split_position = None

        max_attempts = 5 # the max number of _ to search for (i.e. subj_date = 1, subj1_subj2_date = 2, etc.)
        for attempt in range(1, max_attempts + 1):
            # Try joining increasing numbers of parts for ID
            for i in range(attempt, len(parts)):

                # ID is composed of parts up to split position minus 1
                potential_id = '_'.join(parts[:i])
                # Date is the next part after ID
                potential_date = parts[i]
                
                # Check if both ID and date match
                id_match = potential_id in df[dset['id_col']].values.astype(str)
                if dset['date_col'] is not None:
                    date_match = potential_date in df[dset['date_col']].values.astype(str)

                # Check if no valid date, in which case don't consider it
                if potential_date == '00000000': date_match = True
                
                if id_match and date_match:
                    split_position = i+1 # id_date
                        
        if not split_position:
            raise Exception('Error: Aberrant relationship between raw file names and metadata IDs/dates')
        
        # Extract subject-date combinations and collect associated files
        subject_date_files = {}
        
        for f in raw_files:
            basename = os.path.basename(f)
            parts = basename.split('_')
            subj_date = '_'.join(parts[:split_position])
            
            if subj_date not in subject_date_files:
                subject_date_files[subj_date] = []
            
            for suffix in file_suffixes:
                if suffix in basename:
                    subject_date_files[subj_date].append(suffix)
                    break

        # Only keep subjects that have ALL required files   
        to_remove = []
        for key in subject_date_files.keys():
            for suffix in file_suffixes:
                try: assert suffix in subject_date_files[key]
                except AssertionError:
                    to_remove.append(key)
                    break
        for r in to_remove: subject_date_files.pop(r) 
        subj_timepoints = [k for k in subject_date_files.keys()]
        
        # Keep only selected subject groups
        if dset['select'] == 'all':
            pass
        else:
            column, valid_val = dset['select'].split("==", 1)
            valid_val = str(valid_val).strip() # make datatypes equivalent   
            df = df[df[column].astype(str).str.strip() == valid_val] # 

        # Apply age limiting if specified
        if age_filter: # min age
            if age_filter < 0: # if doing [less]
                df = df[df[dset['age_col']] < abs(age_filter)]
            else: # if doing [greater or equal too]
                df = df[df[dset['age_col']] >= age_filter]
            # If no remaining valid subjects, skip this dataset
            if df.empty: continue

        # Filter the DataFrame to include only valid subjects
        if dset['date_col'] is not None: 
            mask = df.apply(lambda row: f"{row[dset['id_col']]}_{row[dset['date_col']]}" in subj_timepoints, axis=1)
        else:
            mask = df.apply(lambda row: f"{row[dset['id_col']]}_00000000" in subj_timepoints, axis=1)
        filtered_df = df[mask]

        # Remove duplicates (i.e. if df has multiple rows for the same participant/timepoint)
        if dset['date_col'] is not None: 
            filtered_df = filtered_df.drop_duplicates(subset=[dset['id_col'], dset['date_col']])
        else:
            filtered_df = filtered_df.drop_duplicates(subset=[dset['id_col']])

        # Compute sex statistics for ratio
        sex_counts = filtered_df[dset['sex_col']].value_counts()
        n_males = sex_counts.get(dset['sex_mapping']['Male'], 0)
        n_females = sex_counts.get(dset['sex_mapping']['Female'], 0)
 
        # Save this to the training set if applicable
        if dset['set'] == 'training' or dset['set'] == 'pretraining':
            for _, row in filtered_df.iterrows():
                training_data.append({
                    'age': row[dset['age_col']]
                })
            training_sex_counts['Female'] += n_females
            training_sex_counts['Male'] += n_males
        
        # Add a new row to d_table for this dataset
        new_row = {
            'repository' : dset['name'],
            'set' : dset['set'],
            'N' : len(filtered_df),
            'min' : f'{filtered_df[dset["age_col"]].min():.1f}',
            'max' : f'{filtered_df[dset["age_col"]].max():.1f}',
            'μ' : f'{filtered_df[dset["age_col"]].mean():.1f}',
            'σ' : f'{filtered_df[dset["age_col"]].std():.1f}',
            'M:F' : f'1 / {n_females/n_males:.1f}'
        }
        d_table = pd.concat([d_table, pd.DataFrame([new_row])], ignore_index=True)

    # Add combined training stats row if we have training sets
    if training_data:

        # Store all possible sex mappings from training datasets
        sex_mappings = {'Female': set(), 'Male': set()}
        for dset in datasets:
            if dset['set'] == 'training' or dset['set'] == 'pretraining':
                sex_mappings['Female'].add(dset['sex_mapping']['Female'])
                sex_mappings['Male'].add(dset['sex_mapping']['Male'])

        # Get the training data as a combined df
        train_df = pd.DataFrame(training_data)

        # Add a new row to d_table for the training datasets
        if dset['set'] == 'pretraining': set_name = 'All Pretraining'
        else: set_name = 'All Training'

        combined_row = {
            'repository' : set_name,
            'set' : 'combined',
            'N' : len(train_df),
            'min' : f'{train_df["age"].min():.1f}',
            'max' : f'{train_df["age"].max():.1f}',
            'μ' : f'{train_df["age"].mean():.1f}',
            'σ' : f'{train_df["age"].std():.1f}',
            'M:F' : f'1 / {training_sex_counts["Female"]/training_sex_counts["Male"]:.1f}'
        }
        d_table = pd.concat([d_table, pd.DataFrame([combined_row])], ignore_index=True)
    
    return d_table

# Function for highlighting target regions. Angle is specified within MATLAB function
def highlight_regions(roi, ico=6, first='rh'):

    # Load in the freesurfer label data
    if ico == 7:
        fsavg_path = '/mnt/md0/softwares/freesurfer/subjects/fsaverage/'
    else:
        fsavg_path = f'/mnt/md0/softwares/freesurfer/subjects/fsaverage{ico}/'

    rh_labels, _, rh_names = nib.freesurfer.read_annot(f'{fsavg_path}label/rh.aparc.a2009s.annot')
    lh_labels, _, lh_names = nib.freesurfer.read_annot(f'{fsavg_path}label/lh.aparc.a2009s.annot')

    # Combine hemispheres with tracking
    if first == 'rh':
        labels = np.hstack((rh_labels, lh_labels + rh_labels.max() + 1))
        names = [(n.decode('utf-8'), 'rh') for n in rh_names] + [(n.decode('utf-8'), 'lh') for n in lh_names]
    else:
        labels = np.hstack((lh_labels, rh_labels + lh_labels.max() + 1))
        names = [(n.decode('utf-8'), 'lh') for n in lh_names] + [(n.decode('utf-8'), 'rh') for n in rh_names]

    # Each unique label has a different value within the masked array (i.e. region 1 = 1, region 2 = 2, etc., all non-regions are 0)
    label_values = np.zeros_like(labels)

    # Find ROI indices and assign unique values
    for roi_idx, roi_name in enumerate(roi, 1):  # Start from 1
        print(f'Region: {roi_name}, i = {roi_idx}')
        # Find matching labels for this ROI (checking both hemispheres)
        roi_label_indices = []
        for i, (name, hemi) in enumerate(names):
            if name == roi_name:
                roi_label_indices.append(i)
        
        # Set the label values for this ROI
        for label_idx in roi_label_indices:
            mask = labels == label_idx
            label_values[mask] = roi_idx

    """
    print(f"Created label array with {len(roi)} ROIs")
    print(f"ROI values: {np.unique(label_values[label_values > 0])}")
    print(f"Total vertices: {len(label_values)}")
    print(f"ROI vertices: {np.sum(label_values > 0)}")
    print(f"Background vertices: {np.sum(label_values == 0)}")
    """

    # Flip hemispheres for nahian_code (which assumes left hemisphere first)
    if first == 'rh':
        assert len(label_values) % 2 == 0  # verify that there are an even number of vertices
        half = len(label_values) // 2  # find the halfway point
        right_hemisphere = label_values[:half]  # select the first half
        left_hemisphere = label_values[half:]  # select the second half
        # Concat with left hemisphere first
        ico_vertices = np.concatenate((left_hemisphere, right_hemisphere))  # swap the halves
    else:  # if lh is already first
        ico_vertices = label_values

    # Save label_values to .mat file for MATLAB
    output_prefix = f'/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/last_model_outputs/all_positive_AD-CN-regions_ico{ico}'
    mat_filename = f'{output_prefix}_labels.mat'

    # Save the label values
    scipy.io.savemat(mat_filename, {'vertex_labels': ico_vertices})
    #print(f"Saved label values to: {mat_filename}")

    # Run the MATLAB code
    matlab_params = f"'{mat_filename}', {ico}, '{output_prefix}'"
    command = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_rois({matlab_params}); exit"]
    result = subprocess.run(command, cwd="/mnt/md0/tempFolder/samAnderson/nahian_code/", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    #print(result)
    
    return

# Function for showing the differences between two sets
def show_ranked_differences(suffix, output_dir):

    # Load data and compute positive regions
    region_stats_df = pd.read_csv(f'{output_dir}{suffix}_errors.csv', index_col=0)

    # Calculate average gap for all regions (not filtering for positive)
    all_regions = (
        region_stats_df.groupby(['freesurfer region', 'hemi'])['age gap'].first()
        .unstack()
        .assign(avg_gap=lambda x: x.mean(axis=1))
        .sort_values('avg_gap', ascending=False)  # Sort by average gap descending
        .reset_index()
    )

    # Reorder columns for clarity
    all_regions = all_regions[[
        'freesurfer region', 'lh', 'rh', 'avg_gap'
    ]]

    # Print formatted results with proper sorting
    print("\nAll regions ranked by average age gap:")
    print("="*85)
    print(f"{'Region':<35} {'Avg Gap':>8} {'LH Gap':>8} {'RH Gap':>8}")
    print("-"*85)
    for _, row in all_regions.iterrows():
        print(
            f"{row['freesurfer region']:<35} "
            f"{row['avg_gap']:>8.2f}"
            f"{row['lh']:>8.2f} "
            f"{row['rh']:>8.2f} "
        )