from paths_and_imports import *

def test_model(X_test, y_test, model, suffix):
    avg_mae, per_node_e, chr_ages, age_gaps, pred_per_vertex = run_model(None, None, X_test, y_test, model=model,
                        batch_size=8, batch_load=8, n_epochs=n_train_epochs, lr=lr, 
                        print_every=print_every, ico_levels=[6, 5, 4], first=first, intra_w=intra_w, 
                        global_w=global_w, weight_decay=weight_decay, feature_scale=1, dropout_levels=dropout_levels)

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

def compare_cohorts(suffix, cohort_pred, cohort_ref, abs_limits=None):  # pred - ref

    # Get the post-processing object
    p = postprocess(suffix=suffix)

    # Load in the unmasked, corrected brain-age gaps
    cohort_pred = np.load(f'{output_dir}{cohort_pred}_corrected_ME_data.npy').squeeze()
    cohort_ref = np.load(f'{output_dir}{cohort_ref}_corrected_ME_data.npy').squeeze()

    # Difference in brain-age gaps
    cohort_diff = cohort_pred - cohort_ref

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
        t_test = ttest_ind(cohort_pred[mask], cohort_ref[mask])
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
    masked_diff = np.where(significant_mask, cohort_diff, 0)

    # Save masked_diff (clip for visualization)
    p.get_matlab(p.clip_outliers(masked_diff, 1, 99), f'{output_dir}{suffix}_corrected_ME_data')

    # Prepare MATLAB command
    matlab_file = f"{output_dir}{suffix}_corrected_ME_data.mat"
    matlab_file_list = f"{{'{matlab_file}'}}"
    
    # Run the MATLAB code
    if abs_limits is not None:  # manual limits vs min and max limits
        command_primary = ["matlab", "-nodisplay", "-nosplash", "-r", 
            f"generate_brain({matlab_file_list}, {{'lat_L','lat_R','med_R','med_L'}}, {abs_limits}, false); exit"] # hide the colorbar
        command_alt = ["matlab", "-nodisplay", "-nosplash", "-r", 
            f"generate_brain({matlab_file_list}, {{'ant','dor','pos','ven'}}, {abs_limits}, true); exit"]
    else:
        command_primary = ["matlab", "-nodisplay", "-nosplash", "-r", 
            f"generate_brain({matlab_file_list}, {{'lat_L','lat_R','med_R','med_L'}}, [], false); exit"]
        command_alt = ["matlab", "-nodisplay", "-nosplash", "-r", 
            f"generate_brain({matlab_file_list}, {{'ant','dor','pos','ven'}}, [], true); exit"]


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
    return    

# Function to bold significant bars
def bold_significant_bars(df, ax, hue_order, special_case=False):
        
    # Get the test names from the axis
    test_names = [tick.get_text() for tick in ax.get_yticklabels()]
    
    # Create a dictionary that relates test and region to pval    
    pmap = df.set_index(['test_n_subjects', 'region'])['adj_pval'].to_dict()
    
    # If special case last y tick and corresponding hue
    add_sig = False
    if special_case:
        
        # Determine if special case is significant
        if pmap.get((test_names[-1], hue_order[-1]), 1.0) < 0.05:
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
            if pmap.get((test, region), 1.0) < 0.05:
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