from paths_and_imports import *

def test_model(X_test, y_test, model, mask, suffix, test_size=8):

    avg_mae, per_node_e, chr_ages, age_gaps, pred_per_vertex = run_model(None, None, X_test, y_test, model=model,
                        mask=mask, batch_size=test_size, batch_load=test_size, n_epochs=1, lr=lr, 
                        print_every=print_every, ico_levels=[6, 5, 4], first=first, intra_w=intra_w, 
                        global_w=global_w, weight_decay=weight_decay, feature_scale=1, 
                        dropout_levels=dropout_levels, verbose=False)

    # Save the outputted values
    np.save(f'{output_dir}{suffix}_avg_mae.npy', avg_mae)
    np.save(f'{output_dir}{suffix}_per_node_e', per_node_e)
    np.save(f'{output_dir}{suffix}_chr_ages', chr_ages)
    np.save(f'{output_dir}{suffix}_age_gaps', age_gaps)
    np.save(f'{output_dir}{suffix}_pred_per_vertex', pred_per_vertex)
    
    print('\n')
    return
    
def postprocess_model(suffix, factors=None, abs_limits=None, global_limits=20):
    
    # Load the values
    chr_ages = np.load(f'{output_dir}{suffix}_chr_ages.npy')
    age_gaps = np.load(f'{output_dir}{suffix}_age_gaps.npy')
    pred_per_vertex = np.load(f'{output_dir}{suffix}_pred_per_vertex.npy')

    # Run post-processing
    p = postprocess(suffix=suffix)
    
    if factors is None: # get new factors and save them
        plot_paths, region_stats_df, processed_pred_per_vertex, factors = p(chr_ages, age_gaps, pred_per_vertex, abs_limits=abs_limits, global_limits=global_limits)
        np.save(f'{output_dir}{suffix}_factors', factors)
    else: # use the inputted factors
        plot_paths, region_stats_df, processed_pred_per_vertex, _ = p(chr_ages, age_gaps, pred_per_vertex, factors, abs_limits=abs_limits, global_limits=global_limits)
        
    np.save(f'{output_dir}{suffix}_processed_pred_per_vertex', processed_pred_per_vertex) # Save the processed vertices
    del p

    # Plot the generated images
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()

    # Iterate over the images and their corresponding axes
    for ax, path in zip(axes, plot_paths):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis('off')  # Turn off axis labels and ticks

    # Save the df
    region_stats_df.to_csv(f'{output_dir}{suffix}_age_gaps.csv', index=True)

    plt.tight_layout()  # Adjust layout to prevent overlap
    return

def compare_cohorts(suffix, cohort_pred, cohort_ref, mask=None, mask_split=None, abs_limits=None, n_bootstrap=500):  # pred - ref
    
    # Define the postprocessing object
    p = postprocess(suffix=suffix)

    # Load in the CAs    
    CA_cohort_pred = np.load(f'{output_dir}{cohort_pred}_chr_ages.npy').squeeze()
    CA_cohort_ref = np.load(f'{output_dir}{cohort_ref}_chr_ages.npy').squeeze()

    # Apply masking
    if mask is not None: 
        CA_cohort_pred = CA_cohort_pred[mask == mask_split['pred']]
        CA_cohort_ref = CA_cohort_ref[mask == mask_split['ref']]

    # Determine if the CAs are significantly different
    _, pval = ttest_ind(CA_cohort_pred, CA_cohort_ref)
    if pval < 0.05:

        print(f'Significant (p = {pval:.2f}) differences in CA distribution across cohorts detected, performing bootstrapping...')

        # Define lists to store bootstrapped results        
        all_ME_pred = []
        all_ME_ref = []

        # Load per-vertex predictions
        pred_per_vertex_cohort_pred = np.load(f'{output_dir}{cohort_pred}_processed_pred_per_vertex.npy')
        pred_per_vertex_cohort_ref = np.load(f'{output_dir}{cohort_ref}_processed_pred_per_vertex.npy')

        # Match CA distributions via 1-year bins
        min_age = int(np.floor(min(CA_cohort_pred.min(), CA_cohort_ref.min())))
        max_age = int(np.ceil(max(CA_cohort_pred.max(), CA_cohort_ref.max())))
        bins = np.arange(min_age, max_age + 1)  # 1-year bins

        for _ in range(n_bootstrap):

            pred_indices = []
            ref_indices = []

            for i in range(len(bins) - 1):
                pred_bin = np.where((CA_cohort_pred >= bins[i]) & (CA_cohort_pred < bins[i+1]))[0]
                ref_bin = np.where((CA_cohort_ref >= bins[i]) & (CA_cohort_ref < bins[i+1]))[0]
                n = min(len(pred_bin), len(ref_bin))

                if n > 0:
                    pred_indices.extend(np.random.choice(pred_bin, n, replace=False))
                    ref_indices.extend(np.random.choice(ref_bin, n, replace=False))

            # Subset matched samples
            matched_pred = pred_per_vertex_cohort_pred[pred_indices, :]
            matched_ref = pred_per_vertex_cohort_ref[ref_indices, :]

            # Compute mean error (per vertex)
            ME_pred = (matched_pred - CA_cohort_pred[pred_indices][:, np.newaxis]).mean(axis=0)
            ME_ref = (matched_ref - CA_cohort_ref[ref_indices][:, np.newaxis]).mean(axis=0)

            all_ME_pred.append(ME_pred)
            all_ME_ref.append(ME_ref)

        # Final bootstrapped ME: average across iterations
        ME_cohort_pred = np.mean(all_ME_pred, axis=0)
        ME_cohort_ref = np.mean(all_ME_ref, axis=0)

    else:
        # Load precomputed ME data
        ME_cohort_pred = np.load(f'{output_dir}{cohort_pred}_corrected_ME_data.npy').squeeze()
        ME_cohort_ref = np.load(f'{output_dir}{cohort_ref}_corrected_ME_data.npy').squeeze()
            
    # Difference in brain-age gaps
    cohort_diff = ME_cohort_pred - ME_cohort_ref

    # Save this difference
    np.save(f'{output_dir}{suffix}_processed_ME_data', cohort_diff); # not 'corrected' because it may also include bootstrapping

    # Get region labels
    labels, names, ctab = p.get_labels()
    unique_labels = np.unique(labels)

    # Collect stats
    region_stats_df = []
    for label_id in unique_labels:
        if label_id == 0:
            continue  # Skip medial wall

        mask = labels == label_id
        region_name, hemi = names[label_id]

        # Conduct the t-test and get the age gap for the region
        t_test = ttest_ind(ME_cohort_pred[mask], ME_cohort_ref[mask])
        regional_age_gap = np.mean(cohort_diff[mask])

        # Update the df
        region_stats_df.append({
            'label_id': label_id,
            'region': region_name,
            'hemi' : hemi,
            'age_gap' : regional_age_gap,
            't_stat': t_test.statistic,
            'raw_pval': t_test.pvalue
        })

    # Create dataframe
    region_stats_df = pd.DataFrame(region_stats_df)

    # Compute per-region average age gap
    region_avg_map = region_stats_df.groupby('region')['age_gap'].mean()
    region_stats_df['region_avg'] = region_stats_df['region'].map(region_avg_map)

    # Adjust p-values
    adj = multipletests(region_stats_df['raw_pval'], method='fdr_bh')
    region_stats_df['adj_pval'] = adj[1]
    region_stats_df['significant'] = adj[0].astype(int)

    # Rank the regions by age gap
    sig_df = region_stats_df[region_stats_df['adj_pval'] < 0.05].copy()
    sig_df = sig_df.sort_values(by='age_gap', key=lambda x: x.abs(), ascending=False)

    # Build mask
    significant_mask = np.zeros_like(labels, dtype=bool)
    for _, row in sig_df.iterrows():
        if row['significant']:
            significant_mask[labels == row['label_id']] = True
           
    # Remove label_id
    sig_df = sig_df.drop(columns=['label_id'])
    
    # Print out the largest age gaps
    print('\nTop 10 significant age gaps:\n')
    print(sig_df.head(10).to_string(index=False))
    
    # Save the df
    region_stats_df.to_csv(f'{output_dir}{suffix}_age_gaps.csv')

    # Mask the brain-age gap difference array
    masked_diff = np.where(significant_mask, cohort_diff, 0).astype(np.float64)

    # Save masked_diff as a MATLAB array, clipping if helpful
    if min(p.clip_outliers(masked_diff, 1, 99)) == max(p.clip_outliers(masked_diff, 1, 99)):
        if min(masked_diff) == max(masked_diff):
            print('No significant regions found')
            return ME_cohort_pred, ME_cohort_ref
        else:
            # Save masked_diff
            p.get_matlab(masked_diff, f'{output_dir}{suffix}_corrected_ME_data')
    else:
        # Save masked_diff (clip for visualization)
        p.get_matlab(p.clip_outliers(masked_diff, 1, 99), f'{output_dir}{suffix}_corrected_ME_data')

    # Prepare MATLAB command
    matlab_file = f'{output_dir}{suffix}_corrected_ME_data.mat'
    matlab_file_list = f"{{'{matlab_file}'}}"
    
    # Run the MATLAB code
    if abs_limits is not None:  # manual limits vs min and max limits
        command_primary = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'lat_L','lat_R','med_R','med_L'}}, {abs_limits}, false); exit"]
        command_alt = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'ant','dor','pos','ven'}}, {abs_limits}); exit"]
    else:
        command_primary = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'lat_L','lat_R','med_R','med_L'}}, [], false); exit"]
        command_alt = ["matlab", "-nodisplay", "-nosplash", "-r", f"generate_brain({matlab_file_list}, {{'ant','dor','pos','ven'}}); exit"]

    result = subprocess.run(command_primary, cwd="/mnt/md0/tempFolder/samAnderson/nahian_code/", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    #print(result)
    result = subprocess.run(command_alt, cwd="/mnt/md0/tempFolder/samAnderson/nahian_code/", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    #print(result)

    # ==== Plot the final images side-by-side ====
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    img1_path = f'{matlab_file[:-4]}_latL_latR_medR_medL.png'
    img2_path = f'{matlab_file[:-4]}_ant_dor_pos_ven.png'
    
    img1 = plt.imread(img1_path)
    axs[0].imshow(img1)
    axs[0].axis('off')

    img2 = plt.imread(img2_path)
    axs[1].imshow(img2)
    axs[1].axis('off')

    plt.tight_layout()

    return ME_cohort_pred, ME_cohort_ref

def bold_significant_bars(df, ax, y_axis, hue, hue_order, pval_thresh = 0.05, special_case=False):
        
    # Get the test names from the axis
    test_names = [tick.get_text() for tick in ax.get_yticklabels()]
    
    # Create a dictionary that relates test and region to pval    
    pmap = df.set_index([y_axis, hue])['adj_pval'].to_dict()
    
    # If special case last y tick and corresponding hue
    add_sig = False
    if special_case:
        
        # Determine if special case is significant
        if pmap.get((test_names[-1], hue_order[-1]), 1.0) < pval_thresh:
            add_sig = True
            
        # Remove the special case
        test_names = test_names[:-1]
        hue_order = hue_order[:-1]

    # Create a list to store sig values in order of the bars
    sig = []

    # Iterate over the regions and tests in the same way the bars are organized
    for region in hue_order:
                
        # Iterate over the cognitive tests
        for test in test_names:
                        
            # Determine if the corresponding test/region combination if significant
            if pmap.get((test, region), 1.0) < pval_thresh:
                sig.append(True)
            else:
                sig.append(False) 
            
    # If the special case was true
    if add_sig: 
        sig.append(True)     
                                                        
    for bar, is_sig in zip(ax.patches, sig[:len(sig)]):
        
        if is_sig:
            # Get bar coordinates
            x, y = bar.get_x(), bar.get_y()
            width, height = bar.get_width(), bar.get_height()

            # Draw a bold outline rectangle on top
            ax.add_patch(plt.Rectangle(
                (x, y), width, height,
                fill=False, linewidth=2.5, edgecolor='black', zorder=10
            ))

            # Also make the bar itself thicker and more visible
            bar.set_linewidth(2)
            bar.set_edgecolor('black')
                                    
    return

def compare_age_gaps(cohort_one, label_one, cohort_two, label_two, x_step=2.5): # CN, AD; M, F
    
    # Set seaborn style
    sns.set_style('whitegrid')

    # Create KDE plot
    plt.figure(figsize=(11, 6))
    sns.kdeplot(cohort_one, color='royalblue', label=label_one, fill=True, alpha=0.4, linewidth=2)
    sns.kdeplot(cohort_two, color='darkorange', label=label_two, fill=True, alpha=0.4, linewidth=2)

    # Add vertical lines at the medians
    plt.axvline(np.median(cohort_one), color='royalblue', linestyle='--', linewidth=2)
    plt.axvline(np.median(cohort_two), color='darkorange', linestyle='--', linewidth=2)

    # Set x-axis ticks centered at 0 with a step of x_ticks, rotated 45 degrees
    xtick_min = np.floor(min(np.min(cohort_one), np.min(cohort_two)) / x_step) * x_step
    xtick_max = np.ceil(max(np.max(cohort_two), np.max(cohort_two)) / x_step) * x_step
    plt.xticks(np.arange(xtick_min, xtick_max + 0.1, x_step), rotation=45, fontsize=15)
    plt.yticks(fontsize=15)

    # Customize plot
    plt.xlabel('Global Age Gap (years)', size=16, labelpad=10)
    plt.ylabel('Density', size=16, labelpad=10)
    plt.legend(fontsize=16)
    plt.tight_layout()
    return plt

def display_top_regions(df, to_rank='age_gap', top=3):

    # Filter the df
    filtered_df = df[df['adj_pval'] < 0.05]

    # Sort by region_avg, remove duplicate regions (pick highest hemisphere)
    top_regions = (
        filtered_df
        .assign(abs_region_avg=lambda d: d['region_avg'].abs())
        .sort_values('abs_region_avg', ascending=False)
        .drop_duplicates('region')
        .head(top)['region']
        .tolist()
    )

    # Select all rows from both hemispheres for those regions
    top_df = df[df['region'].isin(top_regions)]
    # Create new column: 'age_gap_L' or 'age_gap_R' from 'age_gap' and 'hemi'
    top_df[f'{to_rank}_{top_df.hemi.iloc[0]}'] = top_df[to_rank]  # quick hack for warning avoidance
    top_df = top_df.assign(**{
        f'{to_rank}_{hemi}': top_df[top_df['hemi'] == hemi][to_rank]
        for hemi in ['L', 'R']
    })

    # Pivot to get one row per region, with both hemispheres' age gaps
    pivoted = top_df.pivot(index='region', columns='hemi', values=to_rank).reset_index()
    pivoted.columns = ['region', f'{to_rank}_L', f'{to_rank}_R']

    # Merge back region_avg and adj_pval from original filtered_df (just once per region)
    meta = (
        filtered_df[filtered_df['region'].isin(top_regions)]
        .sort_values('region_avg', ascending=False)
        .drop_duplicates('region')[['region', 'region_avg', 'adj_pval']]
    )

    # Final table
    final = pd.merge(pivoted, meta, on='region')

    # Display with desired column order
    column_order = ['region', f'{to_rank}_L', f'{to_rank}_R', 'region_avg', 'adj_pval']
    final = final.reindex(final['region_avg'].abs().sort_values(ascending=False).index)
    final['adj_pval'] = final['adj_pval'].apply(lambda x: f"{x:.2e}")
    display(final[column_order].style.hide(axis='index'))

def build_and_impute_cog_matrix(cog_path, data_dir, cohort_filter=False,
                                min_subject_coverage=0.5, min_test_coverage=0.6,
                                n_bootstrap=100, min_subj=5, min_tests=2,
                                test_relations=False, # correct tests if present
                                tests_to_include=None, # overrides excluded
                                excluded_cols = 
                                    ['FIRSTSCANVISCODE', 'EXAMDATE', 'DOB', 'SITE', 
                                     'FIRSTSCANDATE', 'VISCODE', 'EXAMDATE', 'PTGENDER'
                                     'DAYSSINCEBASELINESCAN', 'DX_BL', 'DX_CURRENT',
                                     'AGE_AT_FIRST_SCAN', 'PTETHCAT', 'PTRACCAT', 
                                     'PTMARRY']):
 
    # Load cognitive scores
    cog = pd.read_csv(cog_path)

    # Create a new column for subject and date combinations
    cog['subject_date'] = (cog['PTID'] + "_" + pd.to_datetime(cog['EXAMDATE'], format='%m/%d/%Y').dt.strftime('%Y%m%d'))
    cog = cog.drop(columns=['PTID'])

    # Filter cog scores based on ADNI cohort (only include those with MRIs)
    if cohort_filter:
        subjects = np.load(f'{data_dir}subj_IDs_ADNI_{cohort_filter}.npy').astype(str)
    else:
        subjects_CN = np.load(f'{data_dir}subj_IDs_ADNI_CN.npy').astype(str)
        subjects_AD = np.load(f'{data_dir}subj_IDs_ADNI_AD.npy').astype(str)
        subjects = np.concatenate([subjects_CN, subjects_AD]); del subjects_CN, subjects_AD
    cog = cog[cog['subject_date'].isin(subjects)]

    # Remove unwanted columns (i.e. not cognitive scores)
    if tests_to_include is None:
        tests_to_include = [col for col in cog.columns if col not in excluded_cols + ['PTID'] + ['subject_date']]
    cog = cog[['subject_date'] + tests_to_include]

    # Clean TRABSCOR if present
    if 'TRABSCOR' in tests_to_include:
        cog.loc[cog['TRABSCOR'] == 300, 'TRABSCOR'] = np.nan

    # Map sex if present
    if 'PTGENDER' in tests_to_include:
        cog['PTGENDER'] = cog['PTGENDER'].map({'Male': 0, 'Female': 1})
        
    # Make it so that increasing scores are correlated with worse performance
    if test_relations:
        for test in cog:
            if test in test_relations.keys():
                if test_relations[test]:
                    cog[test] = -cog[test]

    # Build subject × test matrix
    matrix = cog.drop_duplicates('subject_date').set_index('subject_date')

    # Filter subjects with enough observed tests
    n_tests_initial = len(tests_to_include)
    min_tests_per_subject = int(np.ceil(min_subject_coverage * n_tests_initial))
    matrix = matrix[matrix.notna().sum(axis=1) >= min_tests_per_subject]

    # Filter tests with enough subject coverage
    n_subjects_remaining = len(matrix)
    print(f'Number of included subjects: {n_subjects_remaining}\n')
    min_subjects_per_test = int(np.ceil(min_test_coverage * n_subjects_remaining))
    tests_to_keep = matrix.columns[matrix.notna().sum(axis=0) >= min_subjects_per_test].tolist()
    matrix = matrix[tests_to_keep]

    # Bootstrapped regression imputation
    filled = matrix.copy()
    n_skipped = 0
    for col in filled.columns:

        # Identify tests which are missing values
        missing_idx = filled[col].isna()
        if not missing_idx.any():
            continue
        
        # Iterate over missing values
        for idx in filled[missing_idx].index:
            row = filled.loc[idx]

            # Remove the missing cognitive test (to serve as the dependent variable = y)
            predictors = row.drop(labels=[col])

            # Identify which columns are present for that subject (to use as predictors = X)
            available_cols = predictors.dropna().index.tolist()
            available_rows = filled.dropna(subset=[col] + available_cols)

            if len(available_rows) < min_subj or len(available_cols) < min_tests:
                print('Error: Not enough available data for bootstrapping. Skipping subject.')
                n_skipped+=1
                continue

            # Find all subjects which have all of the tests for both X and y
            estimates = []
            for _ in range(n_bootstrap): # Repeat this process a set number of times
                
                # Randomly select 100 valid subjects, with repeats allowed, to estimate the distribution of the relationship (bootstrapping)
                sample = available_rows.sample(n=len(available_rows), replace=True)
                X_train = sample[available_cols].values
                y_train = sample[col].values

                # Run the regression for the selected datapoints
                model = LinearRegression().fit(X_train, y_train)
                X_pred = predictors[available_cols].values.reshape(1, -1)
                y_pred = model.predict(X_pred)[0]
                estimates.append(y_pred)

            filled.at[idx, col] = np.mean(estimates)

    print(f'n_skipped: {n_skipped}')
    return filled