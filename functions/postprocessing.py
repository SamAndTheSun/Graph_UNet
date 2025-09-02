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

        # Save the corrected age gaps
        np.save(f'{self.output_dir}{self.suffix}_corrected_age_gaps.npy', corrected_age_gap)

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
            command_primary = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'lat_L','lat_R','med_R','med_L'}}, {abs_limits}, false); exit"]
            command_alt = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'ant','dor','pos','ven'}}, {abs_limits}); exit"]
        else:
            command_primary = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'lat_L','lat_R','med_R','med_L'}}, [], false); exit"]
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
    def avg_region_error(self, value_per_vertex_subject, target_region, hemi=None, mean='by_subject'):

        # Ensure labels and names are loaded
        if not hasattr(self, 'labels'):
            self.get_labels()

        mask = np.zeros_like(self.labels, dtype=bool)

        # Handle case: all vertices except medial wall
        if target_region == 'all':
            mask = (self.labels != 0)

        # Handle case: multiple regions + multiple hemis
        elif isinstance(target_region, list):
            if not isinstance(hemi, list) or len(target_region) != len(hemi):
                raise ValueError("If target_region is a list, hemi must be a list of the same length.")

            for region_name, region_hemi in zip(target_region, hemi):
                if region_hemi == 'both':
                    for i, (name, hemi_label) in enumerate(self.names):
                        if region_name.lower() in name.lower() and hemi_label in ['lh', 'rh']:
                            mask |= (self.labels == i)
                else:
                    for i, (name, hemi_label) in enumerate(self.names):
                        if region_name.lower() in name.lower() and region_hemi == hemi_label:
                            mask |= (self.labels == i)

        # Handle case: single region
        else:
            if hemi is None:
                raise Exception('Error: Hemisphere must be included for local error computation')

            if hemi == 'both':
                for i, (name, hemi_label) in enumerate(self.names):
                    if target_region.lower() in name.lower() and hemi_label in ['lh', 'rh']:
                        mask |= (self.labels == i)
            else:
                for i, (name, hemi_label) in enumerate(self.names):
                    if target_region.lower() in name.lower() and hemi == hemi_label:
                        mask |= (self.labels == i)

        # Error if nothing matched
        if not np.any(mask):
            raise ValueError(f"No vertices found for region(s) '{target_region}' with hemi '{hemi}'")

        # Return the requested average
        if mean == 'by_subject':
            return np.mean(value_per_vertex_subject[:, mask], axis=1)
        elif mean == 'by_vertice':
            return np.mean(value_per_vertex_subject[:, mask], axis=0)
        else:
            raise ValueError(f"Unknown mean method: {mean}")
        
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
def regress_cognitive(data_dir, output_dir, cog_path, test_relations,
                      subset=True, regions=None, partial_region_names=False,
                      postprocess_obj=None, covariate_test=None, 
                      get_beta_arrays=False,
                      mask_by='adjusted', pval_thresh=0.05):

    # === Prep the cognitive scores ===
    cognitive_scores = pd.read_csv(cog_path)

    # Combine PTID and scan date
    cognitive_scores['subject_date'] = (
        cognitive_scores['PTID'] + "_" +
        pd.to_datetime(cognitive_scores['EXAMDATE'], format='%m/%d/%Y').dt.strftime('%Y%m%d')
    )

    # Replace TRABSCOR=300 with NaN
    trabscor_mask = cognitive_scores['TRABSCOR'] == 300
    cognitive_scores.loc[trabscor_mask, 'TRABSCOR'] = np.nan    

    # === Load available tests per cohort ===
    all_tests_with_subjects = {}
    for cohort in ['CN', 'AD']:
        subjects = np.load(f'{data_dir}subj_IDs_ADNI_{cohort}.npy').astype(str)
        matched_scores = cognitive_scores[cognitive_scores['subject_date'].isin(subjects)]
        cog_tests = matched_scores.columns.difference([
            'Subject ID', 'Sex', 'Research Group', 'Visit', 'Study Date',
            'Age', 'Modality', 'Description', 'Image ID', 'subject_date'
        ])
        for test in cog_tests:
            key = f'{test}_{cohort}'
            valid_subjects = matched_scores.loc[matched_scores[test].notna(), 'subject_date'].astype(str).tolist()
            all_tests_with_subjects[key] = valid_subjects

    # Load subject pools & ages
    CN_subjects = np.array([s.strip() for s in np.load(f'{data_dir}subj_IDs_ADNI_CN.npy').astype(str)])
    AD_subjects = np.array([s.strip() for s in np.load(f'{data_dir}subj_IDs_ADNI_AD.npy').astype(str)])
    CN_ages = np.load(f'{data_dir}y_ADNI_CN.npy')
    AD_ages = np.load(f'{data_dir}y_ADNI_AD.npy')

    # Map test -> indices
    indices = defaultdict(list)
    CN_series = pd.Series(CN_subjects)
    AD_series = pd.Series(AD_subjects)

    test_to_include = ['ADAS11', 'CDRSB', 'DIGITSCOR', 'EcogPtTotal', 
                       'EcogSPTotal', 'FAQ', 'LDELTOTAL', 'MMSE', 'MOCA', 
                       'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_perc_forgetting', 
                       'TRABSCOR']

    for test_key, subject_dates in all_tests_with_subjects.items():
        if test_key[:-3] not in test_to_include:
            continue
        cohort = test_key[-2:]
        subject_series = CN_series if cohort == 'CN' else AD_series
        mask = subject_series.isin(subject_dates)
        indices[test_key] = mask[mask].index.tolist()

    # === Load brain age gaps ===
    brain_age_gaps = {
        'CN': np.load(f'{output_dir}test_CN_processed_pred_per_vertex.npy') - CN_ages[:, np.newaxis],
        'AD': np.load(f'{output_dir}test_AD_processed_pred_per_vertex.npy') - AD_ages[:, np.newaxis]
    }

    all_results = []
    all_test_scores = {}

    # === Main loop ===
    for test in indices.keys():
        cohort = test[-2:]
        test_name = test[:-3]

        # Match covariate test to this cohort
        covariate_key = f"{covariate_test}_{cohort}" if covariate_test else None

        # Skip covariate test itself
        if covariate_key is not None and test == covariate_key:
            continue

        # Subject pool & ages for this cohort
        subject_list = CN_subjects if cohort == 'CN' else AD_subjects
        all_chr_age = CN_ages if cohort == 'CN' else AD_ages

        # Get base subjects and y
        ordered_subjects = subject_list[indices[test]]
        y = brain_age_gaps[cohort][indices[test]]

        # If covariate test is provided, filter to common subjects
        if covariate_key is not None and covariate_key in indices:
            cov_subjects = subject_list[indices[covariate_key]]
            common_subjects = np.intersect1d(ordered_subjects, cov_subjects)
            mask = np.isin(ordered_subjects, common_subjects)
            y = y[mask]
            ordered_subjects = ordered_subjects[mask]
            chr_age = all_chr_age[indices[test]][mask]
        else:
            chr_age = all_chr_age[indices[test]]

        # Get matching cognitive data
        matched_scores = cognitive_scores[cognitive_scores['subject_date'].isin(ordered_subjects)].copy()
        matched_scores['subject_date'] = pd.Categorical(
            matched_scores['subject_date'], categories=ordered_subjects, ordered=True
        )
        matched_scores = matched_scores.sort_values('subject_date').drop_duplicates('subject_date')

        sex = matched_scores['PTGENDER'].map({'Male': 0, 'Female': 1}).values
        education = matched_scores['PTEDUCAT'].values
        test_scores = matched_scores[test_name].values

        if test_relations[test_name]:
            test_scores = -test_scores

        # Z-score normalization
        education = (education - np.mean(education)) / np.std(education)
        test_scores = (test_scores - np.mean(test_scores)) / np.std(test_scores)

        # Save the test scores
        all_test_scores[test] = test_scores

        # Add covariate scores if present
        if covariate_key is not None and covariate_key in indices:
            cov_name = covariate_key[:-3]
            cov_scores = matched_scores[cov_name].values
            if test_relations[cov_name]:
                cov_scores = -cov_scores
            cov_scores = (cov_scores - np.mean(cov_scores)) / np.std(cov_scores)
            X = np.column_stack((sex, education, test_scores, cov_scores, chr_age))
            X_df = pd.DataFrame(X, columns=['sex', 'education', 'test_score', 'covariate_score', 'chronological_age'])
        else:
            X = np.column_stack((sex, education, test_scores, chr_age))
            X_df = pd.DataFrame(X, columns=['sex', 'education', 'test_score', 'chronological_age'])

        assert len(X_df) == len(y)

        # === Run regressions ===
        if subset:
            if partial_region_names:
                _, names, _ = postprocess_obj.get_labels() # Get all regions
                for partial_region in regions: # Iterate over all partial regions we are interested in
                    matched_regions = []
                    matched_hemispheres = []
                    for full_region, hemi in names: # Iterate over all formal regions
                        if partial_region[0].lower() in full_region.lower(): # Identify if the partial region is in the full region
                            if (partial_region[1] == 'both') or (partial_region[1] == hemi):
                                matched_regions.append(full_region)
                                matched_hemispheres.append(hemi)

                    # Get the average for the entire region
                    y_region = postprocess().avg_region_error(y, matched_regions, matched_hemispheres)
                    results = sm.OLS(y_region, sm.add_constant(X_df)).fit()
                    all_results.append({
                        'cohort': cohort,
                        'test': test_name,
                        'test_n_subjects': f'{test_name}\n(n={len(y_region)})',
                        'region': partial_region[0],
                        'hemi': partial_region[1],
                        'coef': results.params['test_score'],
                        'raw_pval': results.pvalues['test_score'],
                        'r_squared': results.rsquared,
                        'is_inverted': test_relations[test_name]
                    })

            else:
                for region in regions:
                    if region == 'all':
                        y_region = postprocess().avg_region_error(y, region)
                        region_val, hemi_val = region, None
                    elif region[1] == 'both':
                        y_region = postprocess().avg_region_error(y, region[0], 'both')
                        region_val, hemi_val = region[0], region[1]
                    else:
                        y_region = postprocess().avg_region_error(y, region[0], region[1])
                        region_val, hemi_val = region[0], region[1]

                    results = sm.OLS(y_region, sm.add_constant(X_df)).fit()
                    all_results.append({
                        'cohort': cohort,
                        'test': test_name,
                        'test_n_subjects': f'{test_name}\n(n={len(y_region)})',
                        'region': region_val,
                        'hemi': hemi_val,
                        'coef': results.params['test_score'],
                        'raw_pval': results.pvalues['test_score'],
                        'r_squared': results.rsquared,
                        'is_inverted': test_relations[test_name]
                    })
            
        else:
            _, names, _ = postprocess_obj.get_labels()
            for region, hemi in names:
                if region.lower() in ['unknown', 'medialwall']:
                    continue
                y_region = postprocess_obj.avg_region_error(y, region, hemi, mean='by_subject')
                results = sm.OLS(y_region, sm.add_constant(X_df)).fit()
                all_results.append({
                    'cohort': cohort,
                    'test': test_name,
                    'test_n_subjects': f'{test_name}\n(n={len(y_region)})',
                    'region': region,
                    'hemi': hemi,
                    'coef': results.params['test_score'],
                    'raw_pval': results.pvalues['test_score'],
                    'r_squared': results.rsquared,
                    'is_inverted': test_relations[test_name]
                })

    # === Post-processing results ===
    
    # Make it a df
    all_results = pd.DataFrame(all_results)
    
    # Get the region averages (across hemispheres)
    all_results['region_avg'] = (
        all_results.groupby(['cohort', 'region', 'test'])['coef']
        .transform('mean')
    )
    
    # Drop the medial wall if present
    try: all_results = all_results[all_results['region'] != 'Medial_wall']
    except KeyError: pass
    
    # Adjust within each cohort
    all_results['adj_pval'] = (
        all_results.groupby(['cohort'], group_keys=False)['raw_pval']
        .apply(lambda p: pd.Series(
            multipletests(p.values, alpha=0.05, method='fdr_bh')[1],
            index=p.index
        ))
    )

    if not get_beta_arrays:
        return all_results, all_test_scores

    # === Optional: build beta arrays ===
    all_cog_arrays = {}
    labels, names, _ = postprocess().get_labels()
    region_to_label_indices = {(name.lower(), hemi): i for i, (name, hemi) in enumerate(names)}
    tests = all_results['test'].unique()

    for cohort in ['CN', 'AD']:
        cohort_results = all_results[all_results['cohort'] == cohort]
        for test_name in tests:
            test_results = cohort_results[cohort_results['test'] == test_name]
            if (test_results['coef'] == 0).all():
                continue
            region_values = {(row['region'].lower(), row.get('hemi', '')): row['coef']
                             for _, row in test_results.iterrows()}
            display_array = np.zeros_like(labels, dtype=np.float64)
            for (region_name, hemi) in names:
                region_key = (region_name.lower(), hemi)
                if region_key in region_values:

                    # Build the filter mask
                    filter_mask = (
                        (all_results['region'].str.lower() == region_name.lower()) &
                        (all_results['hemi'].str.lower() == hemi.lower()) &
                        (all_results['cohort'].str.lower() == cohort.lower()) &
                        (all_results['test'].str.lower() == test_name.lower())
                    )

                    # Select p-value based on mask_by argument
                    if mask_by == 'adjusted':
                        p = all_results.loc[filter_mask, 'adj_pval'].values[0]
                    elif mask_by == 'raw':
                        p = all_results.loc[filter_mask, 'raw_pval'].values[0]
                    else:
                        p = 0  # No masking

                    # Apply mask if significant
                    if p < pval_thresh:
                        label_index = region_to_label_indices[region_key]
                        display_array[labels == label_index] = region_values[region_key]

                if not np.all(display_array == 0):
                    all_cog_arrays[f'{test_name}_{cohort}'] = display_array

    return all_results, all_cog_arrays, all_test_scores