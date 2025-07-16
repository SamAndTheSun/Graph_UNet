# === Imports === #

# Standard library
import os
import sys
import glob
import subprocess
from collections import defaultdict

# Third-party
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.io
import scipy.sparse as sparse
from scipy.stats import skew, ttest_ind
import re

# Stats and ML
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import Ridge

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Custom paths
sys.path.append('/home/samuelA/.local/lib/python3.10/site-packages')
sys.path.append('/mnt/md0/tempFolder/samAnderson/gnn_model')

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Class for post-processing after model testing
class postprocess():
    def __init__(self, first='rh', suffix='temp',
                 fsavg_path=f'/mnt/md0/softwares/freesurfer/subjects/fsaverage6/', 
                 output_dir='/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/last_model_outputs/'):

        self.first = first
        self.suffix = suffix
        self.fsavg_path = fsavg_path
        self.output_dir = output_dir

    # Get the FreeSurfer labels and names
    def get_labels(self):    
        
        # Get label data        
        rh_labels, rh_ctab, rh_names = nib.freesurfer.read_annot(f'{self.fsavg_path}label/rh.aparc.a2009s.annot')
        lh_labels, lh_ctab, lh_names = nib.freesurfer.read_annot(f'{self.fsavg_path}label/lh.aparc.a2009s.annot')

        # Combine hemispheres with tracking
        if self.first == 'rh':
            labels = np.hstack((rh_labels, lh_labels + rh_labels.max() + 1))
            names = [(n.decode('utf-8'), 'rh') for n in rh_names] + [(n.decode('utf-8'), 'lh') for n in lh_names]
            ctab = np.vstack((rh_ctab, lh_ctab))  # concatenate color tables
        else:
            labels = np.hstack((lh_labels, rh_labels + lh_labels.max() + 1))
            names = [(n.decode('utf-8'), 'lh') for n in lh_names] + [(n.decode('utf-8'), 'rh') for n in rh_names]
            ctab = np.vstack((lh_ctab, rh_ctab))

        self.labels = labels
        self.names = names
        self.ctab = ctab
        
        return labels, names, ctab
    
    # Remove the medial wall from mesh data
    def remove_medial_wall(self, pred_per_vertex):

        if not hasattr(self, 'labels'):
            self.get_labels()

        medial_labels = {'Unknown', 'Medial_wall', '???'}
        medial_indices = [i for i, (name, _) in enumerate(self.names) if name in medial_labels]

        # These are the actual integer label values used in `self.labels`
        medial_label_vals = set(self.ctab[medial_indices, -1])  # last column is the label code

        # Create cortex mask
        cortex_mask = ~np.isin(self.labels, list(medial_label_vals))
        pred_per_vertex_masked = pred_per_vertex[:, cortex_mask]

        return pred_per_vertex_masked, cortex_mask
        
    # Smooth the vertex data; helps to remove model artifact
    def smooth_vertex_data(self, pred_per_vertex, chr_ages, mask, n_iter=4, hops=2):

        _, faces = nib.freesurfer.read_geometry(f'{self.fsavg_path}surf/rh.pial')
        faces = np.vstack((faces, faces + (np.max(faces) + 1)))
        full_n_verts = faces.max() + 1

        if mask is None:
            raise ValueError("Must provide `mask` to remove medial wall influence.")

        # Use only faces that are fully cortical
        valid_faces = np.all(mask[faces], axis=1)
        faces = faces[valid_faces]

        # Reindex vertices to cortex-only indices using fancy indexing
        cortex_indices = np.where(mask)[0]
        index_map = -np.ones(full_n_verts, dtype=int)
        index_map[cortex_indices] = np.arange(cortex_indices.size)
        faces = index_map[faces]  # remap face indices
        n_verts = cortex_indices.size

        # Fast adjacency construction
        row = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        col = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        data = np.ones(len(row), dtype=np.float32)
        adj = sparse.coo_matrix((data, (row, col)), shape=(n_verts, n_verts)).tocsr()
        adj = adj.maximum(adj.T)

        # Build multi-hop smoothing operator using matrix power
        if hops > 1:
            neighborhood = adj.copy()
            for _ in range(hops - 1):
                neighborhood = neighborhood @ adj
            neighborhood = neighborhood + sparse.eye(n_verts)
        else:
            neighborhood = adj + sparse.eye(n_verts)

        # Normalize
        deg = np.array(neighborhood.sum(axis=1)).ravel()
        smoothing_op = sparse.diags(1.0 / deg) @ neighborhood

        # Efficient smoothing loop (still iterative but vectorized)
        smoothed_pred = pred_per_vertex.copy()
        for _ in range(n_iter):
            smoothed_pred = smoothed_pred @ smoothing_op.T

        # Outputs
        vertex_means = np.mean(smoothed_pred, axis=1)
        age_gaps = vertex_means - chr_ages
        per_node_e = np.mean(smoothed_pred - chr_ages[:, None], axis=0)

        return smoothed_pred, age_gaps, per_node_e
        
    # Save plot showing distribution of global age gapes
    def age_gap_plot(self, age_gaps, output_path, min_x=-20, max_x=20):
        
        # Update output path if relevant
        if not output_path.endswith('.png'): output_path += '.png'

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
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved Figure: {output_path}')
        print(f'Figure stats: mean = {np.mean(age_gaps)} ; median = {np.median(age_gaps)} ; std = {np.std(age_gaps)} ; var = {np.var(age_gaps)}')
        plt.clf()

        return
 
    # Account for model bias using correction methods
    def bias_correction(self, chr_ages, pred_per_vertex, factors=None, method='behesti'):

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
        
        return corrected_e, corrected_age_gap, all_corrected, factors
 
    # Clip array outliers based on percentile
    def clip_outliers(self, arr, min_percentile=1, max_percentile=99):
        lower_bound = np.percentile(arr, min_percentile)
        upper_bound = np.percentile(arr, max_percentile)
        return np.clip(arr, lower_bound, upper_bound)
 
    # Convert arrays to matlab format
    def get_matlab(self, per_node_values, output_path=None):
        
        # Update output path if relevant
        if not output_path.endswith('.mat'): output_path += '.mat'
        
        # Put left hemisphere first (assumed in nahian's code)
        if self.first == 'rh':
            assert len(per_node_values) % 2 == 0 # verify that there are an even number of vertices
            half = len(per_node_values) // 2 # find the halfway point
            right_hemisphere = per_node_values[:half] # select the first half
            left_hemisphere = per_node_values[half:] # select the second half
            
            # Concat with left hemisphere first
            ico_vertices = np.concatenate((left_hemisphere, right_hemisphere)) # swap the halves
            
        else: # if lh is already first
            pass
            
        # Save to a .mat file
        scipy.io.savemat(output_path, {'data': ico_vertices})
            
        #print(f"Saved {mat_filename}")
        return
    
    # Get statistics for all regions
    def get_region_stats(self, per_node_e, per_node_e_corrected=None, pred_per_vertex=None, use_abs=True, 
                            remove_medial=True, medial_labels={'Medial_wall', 'Unknown', '???'}):

        # Ensure labels and names are loaded
        if not hasattr(self, 'labels') or not hasattr(self, 'names'):
            self.get_labels()
            
        unique_labels = np.unique(self.labels)
        rows = []

        if pred_per_vertex is None:
            for lid in unique_labels:

                # Mask for the target region
                region_mask = (self.labels == lid)
                region_name, hemi = self.names[lid]
                
                # Skip medial wall
                if remove_medial and region_name in medial_labels: continue
                
                # Get the average error, variance, and skew of the region predictions
                error = per_node_e[region_mask].mean()
                
                rows.append({
                    "region": region_name,
                    "hemi": hemi,
                    "age_gap": f'{error:.2f}',
                    "variance": "-",
                    "skew": "-",
                    "sort_val" : f'{error:.2f}'
                })

        else:
            for lid in unique_labels:

                # Mask for the target region
                region_mask = (self.labels == lid)
                region_name, hemi = self.names[lid]

                # Skip medial wall
                if remove_medial and region_name in medial_labels: continue

                # Get the average error, variance, and skew of the region predictions
                error = per_node_e[region_mask].mean()
                corrected_error = per_node_e_corrected[region_mask].mean()
                var = np.var(pred_per_vertex[:, region_mask], axis=0, ddof=1).mean()
                skew_val = skew(pred_per_vertex[:, region_mask], axis=0).mean()

                rows.append({
                    "region": region_name,
                    "hemi": hemi,
                    "age_gap": f'{corrected_error:.2f} ({error:.2f})',
                    "variance": f'{var:.2f}',
                    "skew": f'{skew_val:.2f}',
                    "sort_val" : corrected_error
                })

        # Create the final DataFrame
        if use_abs: df = pd.DataFrame(rows).sort_values(by="sort_val", key=lambda x: x.abs(), ascending=False)
        else: df = pd.DataFrame(rows).sort_values(by="sort_val", ascending=False)
        
        # Get the average error per-region across hemispheres        
        df['region_avg'] = df.groupby('region')['sort_val'].transform('mean').round(2)
        
        # Remove sort_val
        df = df.drop(columns=['sort_val'])

        return df
    
    # Standard postprocessing line
    def __call__(self, chr_ages, age_gaps, pred_per_vertex, factors=None, use_abs=True, abs_limits=None, global_limits=20):
        '''
        Run basic post-processing, including bias correction, smoothing, and figure generation
        '''

        # Get the global limits for the plot as abs
        global_limits = abs(global_limits)
        
        # Get the anatomic labels
        self.get_labels()

        # Remove the medial wall
        pred_per_vertex, mask = self.remove_medial_wall(pred_per_vertex)

        # Clip the outliers (1st to 99th percentile)
        pred_per_vertex = self.clip_outliers(pred_per_vertex)

        # Smooth the predictions
        pred_per_vertex, age_gaps, smoothed_r_e = self.smooth_vertex_data(pred_per_vertex, chr_ages, mask) # smoothed raw errors

        # Show the distribution of age gaps [Global: BA-CA]
        self.age_gap_plot(age_gaps, output_path=f'{self.output_dir}{self.suffix}_raw_age_gaps', min_x=-global_limits, max_x=global_limits) 

        # Run bias correction
        smoothed_c_e, corrected_age_gap, pred_per_vertex, factors = self.bias_correction(chr_ages, pred_per_vertex, factors) # smoothed corrected errors

        # Show the distribution of age gaps [Global: BA-CA]
        self.age_gap_plot(corrected_age_gap, output_path=f'{self.output_dir}{self.suffix}_corrected_age_gaps', min_x=-global_limits, max_x=global_limits)
        
        # Add back in the medial wall for processing and visualization purposes (to match cortex dims)
        full_r_errors = np.zeros(mask.shape[0], dtype=smoothed_r_e.dtype)
        full_r_errors[mask] = smoothed_r_e; del smoothed_r_e
        #
        full_c_errors = np.zeros(mask.shape[0], dtype=smoothed_c_e.dtype)
        full_c_errors[mask] = smoothed_c_e; del smoothed_c_e
        #
        full_pred_per_vertex = np.zeros((pred_per_vertex.shape[0], mask.shape[0]), dtype=pred_per_vertex.dtype)
        full_pred_per_vertex[:, mask] = pred_per_vertex; del pred_per_vertex

        # Save the error arrays
        np.save(f'{self.output_dir}{self.suffix}_raw_ME_data.npy', full_r_errors) # not clipped, since not for visualization, and not masked
        np.save(f'{self.output_dir}{self.suffix}_corrected_ME_data.npy', full_c_errors) # masked for significance

        # === Determine which regions are significantly different (brain age gap) === #
        
        # Mapping from (region_name, hemi) -> label_id
        region_to_label = {}

        # Create a list to store raw p-values
        raw_pvals = []
        valid_regions = []

        # Iterate over the regions
        for label_id in np.unique(self.labels):
        
            if label_id == 0: continue  # Skip medial wall
            
            # Select for the target region
            mask = self.labels == label_id
            region_name, hemi = self.names[label_id]
            regional_pred = full_pred_per_vertex[:, mask].mean(axis=1)

            # Determine significance and save these results
            t_test = ttest_ind(regional_pred, chr_ages)
            raw_pvals.append(t_test.pvalue)
            valid_regions.append((region_name, hemi))
            region_to_label[(region_name, hemi)] = label_id  # Store for later

        # Correct the p-values
        reject, adj_pval, _, _ = multipletests(raw_pvals, method='fdr_bh')

        # Get the stats by region
        region_stats_df = self.get_region_stats(full_r_errors, full_c_errors, full_pred_per_vertex, use_abs=use_abs)

        # Add the corrected p-values to the dataframe
        pval_map = {(region, hemi): pval for (region, hemi), pval in zip(valid_regions, adj_pval)}
        region_stats_df['adj_pval'] = region_stats_df.apply(
            lambda row: pval_map.get((row['region'], row['hemi']), np.nan), axis=1)
        
        # Rank the regions by age gap
        sig_df = region_stats_df[region_stats_df['adj_pval'] < 0.05].copy()
        sig_df['age_gap_clean'] = sig_df['age_gap'].str.extract(r'([-+]?\d*\.\d+|\d+)')[0].astype(float)
        sig_df = sig_df.sort_values(by='age_gap_clean', key=lambda x: x.abs(), ascending=False)
        sig_df = sig_df.drop(columns=['age_gap_clean'])

        # Print out the largest age gaps
        print('\nTop 10 significant age gaps:\n')
        print(sig_df.head(10).to_string(index=False))
        
        # === Visualization === # 

        # Mask the corrected errors based on significance
        for (region_name, hemi), keep in zip(valid_regions, reject):
            if not keep:
                label_id = region_to_label[(region_name, hemi)]
                mask = self.labels == label_id
                full_c_errors[mask] = 0
        
        # Clip the errors again for visualization purposes, then save the errors as matlab arrays
        self.get_matlab(self.clip_outliers(full_r_errors, 1, 99), output_path=f'{self.output_dir}{self.suffix}_raw_ME_data')
        self.get_matlab(self.clip_outliers(full_c_errors, 1, 99), output_path=f'{self.output_dir}{self.suffix}_corrected_ME_data')

        # Convert to MATLAB cell array syntax
        mat_files = [f'{self.output_dir}{self.suffix}_raw_ME_data', f'{self.output_dir}{self.suffix}_corrected_ME_data']
        matlab_file_list = "{" + ",".join([f"'{f}'" for f in mat_files]) + "}"

        # Run the MATLAB code
        if abs_limits is not None: # manual limits vs min and max limits
            command_primary = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'lat_L','lat_R','med_R','med_L'}}, {abs_limits}); exit"]
            command_alt = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'ant','dor','pos','ven'}}, {abs_limits}); exit"]
        else:
            command_primary = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'lat_L','lat_R','med_R','med_L'}}); exit"]
            command_alt = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'ant','dor','pos','ven'}}); exit"]

        result = subprocess.run(command_primary, cwd="/mnt/md0/tempFolder/samAnderson/nahian_code/", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        #print(result)
        result = subprocess.run(command_alt, cwd="/mnt/md0/tempFolder/samAnderson/nahian_code/", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        #print(result)
        
        # Specify the image paths to load
        paths = [
            f'{self.output_dir}{self.suffix}_raw_ME_data_latL_latR_medR_medL.png', 
            f'{self.output_dir}{self.suffix}_raw_age_gaps.png',
            f'{self.output_dir}{self.suffix}_corrected_ME_data_latL_latR_medR_medL.png',
            f'{self.output_dir}{self.suffix}_corrected_age_gaps.png'
        ]
        
        # Return the paths of the images to plot as a list, and the regional df
        return paths, region_stats_df, full_pred_per_vertex, factors

    # Get the average error for a given region
    def avg_region_error(self, value_per_vertex_subject, target_region, mean='by_subject'):

        # Ensure labels and names are loaded
        if not hasattr(self, 'labels'):
            self.get_labels()

        # Handle full-cortex case
        if target_region == 'all':
            mask = (self.labels != 0) # exclude medial wall
        else:
            mask = np.zeros_like(self.labels, dtype=bool)
            for i, (name, _) in enumerate(self.names):
                if target_region.lower() in name.lower():
                    mask |= (self.labels == i)

        if not np.any(mask):
            raise ValueError(f"No vertices found for region: {target_region}")

        # Average across values within the mask
        if mean == 'by_subject': # so you get 1 value per subject
            return np.mean(value_per_vertex_subject[:, mask], axis=1)
        elif mean == 'by_vertice': # so you get 1 value per vertice
            return np.mean(value_per_vertex_subject[:, mask], axis=0)
    
    # Regress anatomical qualities (from testing data) against age gap
    def regress_region(self, X_test, error_per_vertex, groups_dict=None, target_region='G_oc-temp_med-Parahip', 
                    feature_order=['area','curvature','sulcal_depth', 'thickness', 'white_gray_matter_intensity_ratio']): # used for all files; avoid -
        
        # Ensure labels and names are loaded
        if not hasattr(self, 'labels') or not hasattr(self, 'names'):
            self.get_labels()
        
        # Handle full-cortex case
        if target_region == 'all':
            mask = np.ones_like(self.labels, dtype=bool)
        else:
            # Generate region mask
            if not hasattr(self, 'labels'):
                self.get_labels()
            mask = np.zeros_like(self.labels, dtype=bool)
            for i, (name, _) in enumerate(self.names):
                if target_region.lower() in name.lower():
                    mask |= (self.labels == i)
        
        if not np.any(mask):
            raise ValueError(f"No vertices found for region: {target_region}")
        
        # Compute subject-level averages within region
        X_avg = np.mean(X_test[:, mask, :], axis=1)  # Mean features per subject
        y_avg = self.average_error_within_region(error_per_vertex, target_region)  # Mean error per subject
        
        if groups_dict is None: # standard regression
        
            # Add intercept column
            X_with_intercept = sm.add_constant(X_avg) # adds to front; only needed for sm.OLS not smf.ols
            # Fit the model
            model = sm.OLS(y_avg, X_with_intercept).fit()
                    
        else: # regression with interaction term
            
            # Build DataFrame
            df = pd.DataFrame(X_avg, columns=feature_order)
            df['brain_age'] = y_avg
            # Add group variables to df
            for group, values in groups_dict.items():
                df[group] = values
        
            # For each group, create (feature1 + feature2 + ...) * group term
            morph_str = ' + '.join(feature_order)
            interaction_terms = [f'({morph_str}) * {g}' for g in groups_dict.keys()]
            formula = 'brain_age ~ ' + ' + '.join(interaction_terms)
            
            # Fit the model
            model = smf.ols(formula, data=df).fit()
            
        # Adjust the p-values
        pvals = model.pvalues
        _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
            
        return model, pvals_corrected
        
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

# Function for showing the differences between two sets
def show_ranked_differences(suffix, output_dir):

    # Load data
    region_stats_df = pd.read_csv(f'{output_dir}{suffix}_age_gaps.csv', index_col=0)

    # Remove medial wall and non-informative regions
    medial_labels = {"Medial_wall", "Unknown", "???"}
    region_stats_df = region_stats_df[~region_stats_df["region"].isin(medial_labels)]

    # Extract the corrected (first) numeric value from age_gap string
    def extract_corrected(val):
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group(0)) if match else float('nan')

    region_stats_df['corrected_gap'] = region_stats_df['age_gap'].apply(extract_corrected)

    # Pivot corrected values by hemisphere (lh, rh)
    corrected_pivot = (
        region_stats_df
        .pivot_table(index='region', columns='hemi', values='corrected_gap', aggfunc='first')
        .rename(columns={'lh': 'lh_gap', 'rh': 'rh_gap'})
    )

    # Compute average gap
    corrected_pivot['avg_gap'] = corrected_pivot[['lh_gap', 'rh_gap']].mean(axis=1)

    # Reset index and reorder columns
    all_regions = (
        corrected_pivot
        .reset_index()
        [['region', 'lh_gap', 'rh_gap', 'avg_gap']]
        .sort_values('avg_gap', ascending=False)
    )

    # Print formatted output
    print("\nAll regions ranked by average age gap:")
    print("=" * 85)
    print(f"{'Region':<35} {'Avg Gap':>8} {'LH Gap':>8} {'RH Gap':>8}")
    print("-" * 85)
    for _, row in all_regions.iterrows():
        print(
            f"{row['region']:<35} "
            f"{row['avg_gap']:>8.2f} "
            f"{row['lh_gap']:>8.2f} "
            f"{row['rh_gap']:>8.2f} "
        )

    return

# Function for regressing through cognitive scores
def regress_cognitive(data_dir, output_dir, cog_path='/mnt/md0/tempFolder/samAnderson/datasets/ADNI_cognitive_scores.csv', 
                      subset=True, regions=None, postprocess_obj=None):

    # === Prep the cognitive scores for analysis === #

    # Load cognitive scores
    cognitive_scores = pd.read_csv(cog_path)

    # Combine PTID and scan date
    cognitive_scores['subject_date'] = (
        cognitive_scores['PTID'] + "_" +
        pd.to_datetime(cognitive_scores['EXAMDATE'], format='%m/%d/%Y').dt.strftime('%Y%m%d')
    )

    # Remove values = 300 for TRABSCOR (max time limit, might distort results)
    trabscor_mask = cognitive_scores['TRABSCOR'] == 300 # create a mask
    cognitive_scores.loc[trabscor_mask, 'TRABSCOR'] = np.nan

    # === Custom test modifications === #

    # Load tests
    all_tests_with_subjects = {}

    for cohort in ['CN', 'AD']:
        subjects = np.load(f'{data_dir}subj_IDs_ADNI_{cohort}.npy')
        subjects = subjects.astype(str)  # Ensure string format
        
        matched_scores = cognitive_scores[cognitive_scores['subject_date'].isin(subjects)]
        cog_tests = matched_scores.columns.difference([
            'Subject ID', 'Sex', 'Research Group', 'Visit', 'Study Date',
            'Age', 'Modality', 'Description', 'Image ID', 'subject_date'
        ])

        for test in cog_tests:
            key = f'{test}_{cohort}'
            valid_subjects = matched_scores.loc[matched_scores[test].notna(), 'subject_date'].astype(str).tolist()
            all_tests_with_subjects[key] = valid_subjects

    # Load full subject lists
    CN_subjects = np.load(f'{data_dir}subj_IDs_ADNI_CN.npy')
    CN_subjects = np.array([s.strip() for s in CN_subjects.astype(str)])

    AD_subjects = np.load(f'{data_dir}subj_IDs_ADNI_AD.npy')
    AD_subjects = np.array([s.strip() for s in AD_subjects.astype(str)])

    # Load chronological ages
    CN_ages = np.load(f'{data_dir}y_ADNI_CN.npy')
    AD_ages = np.load(f'{data_dir}y_ADNI_AD.npy')

    # Compute indices for each test and cohort
    indices = defaultdict(list) # dict that defaults to a list

    # Convert subject pools to pandas Series for fast isin checks
    CN_series = pd.Series(CN_subjects)
    AD_series = pd.Series(AD_subjects)

    # Create a dictionary indicating whether higher scores are associated with better performance
    test_relations = {
        'ADAS11' : False,            # Alzheimer's Disease Assessment Scale - Cognitive Subscale (11 items)
        'ADAS13' : False,            # Alzheimer's Disease Assessment Scale - 13-item version
        'ADASQ4' : False,            # Subcomponent of ADAS
        'CDRSB' : False,             # Clinical Dementia Rating - Sum of Boxes
        'DIGITSCOR' : True,          # Digit Span (forward/backward) - higher = better
        'EcogPtDivatt' : False,      # ECog Patient: Divided Attention - higher = more impairment
        'EcogPtLang' : False,        # ECog Patient: Language
        'EcogPtMem' : False,         # ECog Patient: Memory
        'EcogPtOrgan' : False,       # ECog Patient: Organization
        'EcogPtPlan' : False,        # ECog Patient: Planning
        'EcogPtVisspat' : False,     # ECog Patient: Visuospatial
        'EcogPtTotal': False,  # ECog Patient Total Score – higher values indicate greater self-reported cognitive impairment
        'EcogSPDivatt' : False,      # ECog Study Partner: Divided Attention
        'EcogSPLang' : False,        # ECog Study Partner: Language
        'EcogSPMem' : False,         # ECog Study Partner: Memory
        'EcogSPOrgan' : False,       # ECog Study Partner: Organization
        'EcogSPPlan' : False,        # ECog Study Partner: Planning
        'EcogSPTotal' : False,       # ECog Study Partner: Total Score
        'EcogSPVisspat' : False,     # ECog Study Partner: Visuospatial
        'FAQ' : False,               # Functional Activities Questionnaire - higher = more impairment
        'LDELTOTAL' : True,          # Logical Memory Delayed Recall - higher = better
        'MMSE' : True,               # Mini-Mental State Examination
        'MOCA' : True,               # Montreal Cognitive Assessment
        'RAVLT_forgetting' : False,  # Rey Auditory Verbal Learning Test - higher = worse retention
        'RAVLT_immediate' : True,    # RAVLT immediate recall
        'RAVLT_learning' : True,     # RAVLT learning score
        'RAVLT_perc_forgetting' : False,  # Percent forgetting - higher = worse
        'TRABSCOR' : False           # Trail Making Test Part B - Score = time, higher = worse
    }

    # Determine which tests to include
    test_to_include = ['ADAS11', 'CDRSB', 'DIGITSCOR', 'EcogPtTotal', 
                    'EcogSPTotal', 'FAQ', 'LDELTOTAL', 'MMSE', 'MOCA', 
                    'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_perc_forgetting', 
                    'TRABSCOR']

    for test_key, subject_dates in all_tests_with_subjects.items():

        # Skip excluded tests and irrelevant columns
        if test_key[:-3] not in test_to_include: continue

        # Isolate the cohort
        cohort = test_key[-2:]
        
        # Choose appropriate subject pool
        subject_series = CN_series if cohort == 'CN' else AD_series
        
        # Compute boolean mask for matching subjects
        mask = subject_series.isin(subject_dates)
        
        # Save indices of matched subjects
        indices[test_key] = mask[mask].index.tolist()

    # ======= Iterate over the cohorts and match the brain age gaps to the participant cognitive scores ======= #

    # Create a list to store all results for all regions and cognitive tests
    all_results = []

    # Get the brain-age gaps for the AD and CN subjects
    brain_age_gaps = {}
    brain_age_gaps['CN'] = np.load(f'{output_dir}test_CN_processed_pred_per_vertex.npy') - CN_ages[:, np.newaxis]
    brain_age_gaps['AD'] = np.load(f'{output_dir}test_AD_processed_pred_per_vertex.npy') - AD_ages[:, np.newaxis]

    # Loop over the cognitive tests
    for test in indices.keys():
        
        # Determine cohort from test name
        cohort = test[-2:]  # 'CN' or 'AD'
        test_name = test[:-3]  # e.g., 'MMSE' from 'MMSE_CN'

        # Get the corresponding brain_age_gap data
        subject_list = CN_subjects if cohort == 'CN' else AD_subjects
        y = brain_age_gaps[cohort][indices[test]]

        # Get cognitive score data for these specific subjects (using subject_list[indices[test]])
        ordered_subjects = subject_list[indices[test]]  # These are the subjects X is based on
        matched_scores = cognitive_scores[cognitive_scores['subject_date'].isin(ordered_subjects)].copy()

        # Make sure the subject_date is ordered the same as X
        matched_scores['subject_date'] = pd.Categorical(
            matched_scores['subject_date'], categories=ordered_subjects, ordered=True
        )
        matched_scores = matched_scores.sort_values('subject_date')
        
        # Remove duplicates (different scans with same metadata)
        matched_scores = matched_scores.drop_duplicates('subject_date')

        # Get the independent variables of the subjects
        chr_age = CN_ages[indices[test]] if cohort == 'CN' else AD_ages[indices[test]] # aligns with brain_age_gaps
        sex = matched_scores['PTGENDER'].map({'Male': 0, 'Female': 1}).values
        education = matched_scores['PTEDUCAT'].values
        test_scores = matched_scores[test_name].values

        # Invert the sign of the test scores if higher scores are associated with better performance
        if test_relations[test[:-3]]: test_scores = -test_scores

        # Z-score normalize the test scores and education (sex is categorical)
        education = (education - np.mean(education)) / np.std(education)
        test_scores = (test_scores - np.mean(test_scores)) / np.std(test_scores)

        # Create the collective X array
        X = np.column_stack((sex, education, test_scores, chr_age))
        X_df = pd.DataFrame(X, columns=['sex', 'education', 'test_score', 'chronological_age'])

        assert len(X_df) == len(y) # verify shapes

        # ======= Run regressions ======= #

        if subset: # If using specific regions
            
            for region in regions.keys():

                # Average the brain-age predictions across the vertices of the associated region
                y = postprocess().avg_region_error(y, region)
                    
                # Regress the cognitive scores with BA, CA, sex, and education
                cognitive_regression = sm.OLS(y, sm.add_constant(X_df)) # need to manually add constant for statsmodels
                results = cognitive_regression.fit()
                
                # Append the results
                all_results.append({
                    'cohort': test[-2:],
                    'test' : test[:-3],
                    'test_n_subjects': f'{test[:-3]}\n(n={len(y)})',
                    'region': region,
                    'coef': results.params['test_score'],
                    'raw_pval': results.pvalues['test_score'],
                    'is_inverted': test_relations[test[:-3]]
                })

        else: # If using every region

            _, names, _ = postprocess_obj.get_labels() # get the labels, etc.
            for region_name, hemisphere in names:

                if region_name.lower() in ['unknown', 'medialwall']:
                    continue

                # Average across the region, then regress
                y_region = postprocess_obj.avg_region_error(y, region_name, mean='by_subject')
                cognitive_regression = sm.OLS(y_region, sm.add_constant(X_df))
                results = cognitive_regression.fit()

                # Append the results
                all_results.append({
                    'cohort': cohort,
                    'test': test_name,
                    'test_n_subjects': f'{test_name}\n(n={len(y_region)})',
                    'region': region_name,
                    'hemi': hemisphere,
                    'coef': results.params['test_score'],
                    'raw_pval': results.pvalues['test_score'],
                    'is_inverted': test_relations[test_name]
                })


    # Make the results into a dataframe
    all_results = pd.DataFrame(all_results)

    if regions is not None:
        # Replace the regions with their colloquial names
        all_results['region'] = all_results['region'].map(regions).fillna(all_results['region'])

    # Adjust p-values per-test and per-cohort
    all_results['adj_pval'] = (
        all_results
        .groupby(['test', 'cohort'])['raw_pval']
        .transform(lambda p: multipletests(p, alpha=0.05, method='fdr_bh')[1]))

    return all_results
