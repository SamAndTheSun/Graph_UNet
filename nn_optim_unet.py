# Standard library imports
import os
import sys
import glob

# Custom paths
sys.path.append('/home/samuelA/.local/lib/python3.10/site-packages')
sys.path.append('/mnt/md0/tempFolder/samAnderson/gnn_model')

# Third-party imports
import numpy as np
import nibabel as nib
import time
import random
from copy import deepcopy

# Statistical third-party imports
from sklearn.model_selection import KFold

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

# PYG imports
from torch_geometric.data import Data as Data_pyg
from torch_geometric.loader import DataLoader as DataLoader_pyg
from torch_geometric.nn import GCNConv, BatchNorm

###
SEED = 808
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#device = torch.device("cpu")
###

# Set the random seed for reproduceability
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)

# Basic function for running the model
def run_model(X_train, y_train, X_test, y_test, model=None,
              batch_size=32, batch_load=1, n_epochs=100, lr=0.0001, print_every=10, 
              pooling_path='/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/pooling/',
              ico_levels=[6, 5, 4], criterion='variance_and_mae', weight_decay=0.01, 
              first='rh', intra_w=0, global_w=0, feature_scale=1, dropout_levels=[0.5, 0.5]):
    
    # Set seed for reproducability
    set_seed(SEED)
    
    # Print out model hyperparameters
    print(f'''
    === Model/Training Params ===\n
    batch_size = {batch_size}
    batch_load = {batch_load}
    n_epochs = {n_epochs}
    lr = {lr}
    L2 regularization = {weight_decay}
    intra_w = {intra_w}
    global_w = {global_w}
    feature_scale = {feature_scale}
    dropout_levels = {dropout_levels}
    ''')


    # Sort the ico levels in descending order (largest first)
    ico_levels = sorted(ico_levels, reverse=True)

    # Use MAE if specified
    if criterion == 'mae': criterion = F.l1_loss

    # Get the GNN model then train it
    if model == None:
        model = get_gnn(fs=feature_scale, dropout_levels=dropout_levels, pooling_path=pooling_path, ico_levels=ico_levels)
    nn_trainer = nn_builder(model)

    # If training a model but not testing it
    if X_test == None:
        trained_model = nn_trainer(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model=model,
            batch_size=batch_size, batch_load=batch_load, n_epochs=n_epochs, 
            lr=lr, print_every=print_every, ico_levels=ico_levels, 
            criterion=criterion, weight_decay=weight_decay, 
            first=first, intra_w=intra_w, global_w=global_w)
        return trained_model
    
    # If testing a model, 5-fold cv, or training then testing
    else:
        avg_mae, per_node_e, chr_ages, age_gaps, pred_per_vertex = nn_trainer(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model=model,
            batch_size=batch_size, batch_load=batch_load, n_epochs=n_epochs, 
            lr=lr, print_every=print_every, ico_levels=ico_levels, criterion=criterion, 
            weight_decay=weight_decay, first=first, intra_w=intra_w, global_w=global_w)
        return avg_mae, per_node_e, chr_ages, age_gaps, pred_per_vertex

# Define the GNN dynamically
def get_gnn(fs=1, dropout_levels = [0.5, 0.5],
            pooling_path='/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/pooling/', 
            ico_levels=[6, 5, 4]):
    
    # Set seed for reproducability
    set_seed(SEED) 

    # Load in the downsampling indices associating receptive fields across ico levels
    indice_paths = glob.glob(f'{pooling_path}*')
    downsample_indices = {}
    for path in indice_paths:
        target_ico = path[path.find("->")-1]
        if 'downsample' in path:
            # load the np array and convert to tensor, then move to GPU
            downsample_indices[int(target_ico)] = torch.from_numpy(np.load(path)).to(device)

    # Get the edge indices for each ico level
    edge_indices = {}
    for ico in ico_levels:
        # Start by getting the faces
        if ico == 7:
            fsavg_path = '/mnt/md0/softwares/freesurfer/subjects/fsaverage/'
        else:
            fsavg_path = f'/mnt/md0/softwares/freesurfer/subjects/fsaverage{ico}/'
        _, faces = nib.freesurfer.read_geometry(f'{fsavg_path}surf/rh.pial')

        # Stack the faces to account for both hemispheres (same for each hemi)
        faces_both_hemi = np.vstack((faces, faces + (np.max(faces) + 1)))
        # Derive edges from the faces
        edges = np.vstack([
            faces_both_hemi[:, [0, 1]],  # edge 1: v1, v2
            faces_both_hemi[:, [1, 2]],  # edge 2: v2, v3
            faces_both_hemi[:, [2, 0]],  # edge 3: v3, v1
            faces_both_hemi[:, [1, 0]],  # reverse of edge 1
            faces_both_hemi[:, [2, 1]],  # reverse of edge 2
            faces_both_hemi[:, [0, 2]]   # reverse of edge 3
        ])
        # Sort the edges, remove duplicates, and transpose
        sorted_edges = np.sort(edges, axis=1)
        unique_edges = np.unique(sorted_edges, axis=0)
        # Convert to tensor and move to GPU
        edge_indices[ico] = torch.tensor(unique_edges.T, dtype=torch.long, device=device)

    # Create the dict holding the upsample indices
    upsampling_toolkit = {}
    for ico in ico_levels[1:]:  # Skip the largest ico
        # Create tensors relating across receptive fields in reverse (i.e. from lower to higher)
        # Since there are only at most two receptive fields that a higher-ico vertex is a part of, we only need two arrays
        first_coor = torch.full((torch.max(downsample_indices[ico + 1]) + 1, 1), -1, dtype=torch.long, device=device)
        second_coor = torch.full((torch.max(downsample_indices[ico + 1]) + 1, 1), -1, dtype=torch.long, device=device)
        # Loop through the rows of the downsampled indices and get the upsample indices
        # Note that rows can also be thought of as receptive fields
        for row_idx, row in enumerate(downsample_indices[ico + 1]):  # Lower ico nodes
            for indice in row:  # Higher ico nodes
                # Set the index, as contained in the tensor, to the receptive field, as conveyed by the row index
                # so if '100' is in the 3rd row you set index 100 = 3
                if first_coor[indice.item()] == -1:
                    first_coor[indice.item()] = row_idx
                else:
                    second_coor[indice.item()] = row_idx  # Some nodes are represented twice in the array, because they are part of 2 receptive fields

        # Save the relevant list of tensors to the ico dict entry
        upsampling_toolkit[ico] = [first_coor.squeeze(), second_coor.squeeze()]

    # Define the gnn model
    class gnn_model(torch.nn.Module):
        def __init__(self):
            super().__init__()

            # Save the relevant indices
            self.ico_levels = ico_levels
            #
            self.edge_indices = edge_indices
            self.downsample_indices = downsample_indices
            self.upsampling_toolkit = upsampling_toolkit
            #
            self.backups_saved = False # note that we haven't saved backups for batch processing
            #
            self.batch_processed = True # whether or not the arrays have been modified for batch size
            self.last_batch_size = 1 # helps to account for changes in batch size, relevant for preprocessing; defaults to 1                

            ### first block ###
            self.gcn1 = GCNConv(5, 8*fs, cached=True)
            self.bn1 = BatchNorm(8*fs) # ico 6: 16 features
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(p=dropout_levels[0])

            ### second block ###
            self.gcn2 = GCNConv(8*fs, 16*fs, cached=True)
            self.bn2 = BatchNorm(16*fs) # ico 5: 32 feature
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(p=dropout_levels[1])

            ### third block ###
            self.gcn3 = GCNConv(16*fs, 16*fs, cached=True)
            self.bn3 = BatchNorm(16*fs) # ico 4: 32 features
            self.relu3 = nn.ReLU()
            self.dropout3 = nn.Dropout(p=dropout_levels[2])

            ### fourth block ###
            # +32 from skip connection, so 64 features
            self.gcn4 = GCNConv(32*fs, 8*fs, cached=True)
            self.bn4 = BatchNorm(8*fs) # ico 5: 16 features
            self.relu4 = nn.ReLU() 
            self.dropout4 = nn.Dropout(p=dropout_levels[3])

            ### fifth block ###
            # +16 from skip connection, so 32 features
            self.gcn5 = GCNConv(16*fs, 8*fs, cached=True)
            self.bn5 = BatchNorm(8*fs)
            self.relu5 = nn.ReLU() 
            self.dropout5 = nn.Dropout(p=dropout_levels[4])

            ### sixth block ###
            self.gcn6 = GCNConv(8*fs, 1) # output brain

            # save gcn layers in a list for clearing cache later
            self.gcn_layers = [self.gcn1, self.gcn2, self.gcn3, self.gcn4, self.gcn5, self.gcn6]
        
        def downsample_block(self, x_in, ico, weights='mean'):
            # Define the lower ico with respect to its higher-ico receptive field
            x_out = x_in[self.downsample_indices[ico]]
            if weights == 'mean': # Mean pool; avg by receptive field
                x_out = torch.mean(x_out, dim=1)
            elif weights == 'max': # Max pool; max of receptive field
                x_out, _ = torch.max(x_out, dim=1)
            else: # Learnable attn weights
                x_out = torch.sum(F.softmax(weights, dim=1).repeat(self.last_batch_size, 1, x_out.shape[2]) * x_out, dim=1) # softmax(weights) -> attn weights -> learnable weighted sum
            return x_out

        def upsample_block(self, x_in, ico):
            # Get the precomputed coordinates
            first_coor, second_coor = self.upsampling_toolkit[ico]  # shapes: (n_vertices,)

            # Initialize the output tensor
            x_out = torch.zeros((torch.max(self.edge_indices[ico+1]) + 1, x_in.shape[1]), device=device)

            # Use first_coor to identify targets
            x_out[:] = x_in[first_coor]

            # Mask for valid indices in second_coor, add these to the output, then average by the number of receptive fields
            valid_second_coor = second_coor >= 0
            x_out[valid_second_coor] += x_in[second_coor[valid_second_coor]] # combine the values derived from the coordinates
            x_out[valid_second_coor] /= 2 # average the values if two
            return x_out
        
        def batch_process(self, n_batches):

            if not self.backups_saved:
                # Save backups for the indices, so that when batch processing you have them; on the GPU
                self.backups = [
                    {k: v.clone().to(device) for k, v in self.edge_indices.items()},
                    {k: v.clone().to(device) for k, v in self.downsample_indices.items()},
                    {k: [v[0].clone().to(device), v[1].clone().to(device)] for k, v in self.upsampling_toolkit.items()}
                ]
                # note that we have backups saved and don't need to save them again
                self.backups_saved = True

            # Use backups so we aren't influenced by prior preprocessing
            self.edge_indices = {k: v.clone().to(device) for k, v in self.backups[0].items()}
            self.downsample_indices = {k: v.clone().to(device) for k, v in self.backups[1].items()}
            self.upsampling_toolkit = {k: [v[0].clone().to(device), v[1].clone().to(device)] for k, v in self.backups[2].items()}

            # Format into dict for batch processing
            absolute_indices = {
                'edge_indices': self.edge_indices,
                'downsample_indices': self.downsample_indices,
                'upsampling_toolkit': self.upsampling_toolkit
            }

            # Extend indices with respect to batch size
            for indice_type, indices_dict in absolute_indices.items():
                if indice_type == 'upsampling_toolkit':
                    # Handle upsampling indices separately
                    for ico, (first_coor, second_coor) in indices_dict.items():
                        # Extend the coordinates with respect to batch
                        shift = torch.max(first_coor) + 1
                        extended_first_coor = [first_coor]
                        extended_second_coor = [second_coor]
                        # Extend both first_coor and second_coor in a single loop
                        for i in range(1, n_batches):
                            extended_first_coor.append(first_coor + (shift * i))
                            shifted_second_coor = second_coor.clone()
                            shifted_second_coor[shifted_second_coor != -1] += (shift * i)
                            extended_second_coor.append(shifted_second_coor)
                        # Concatenate the extended coordinates
                        extended_first_coor = torch.cat(extended_first_coor, dim=0).long()
                        extended_second_coor = torch.cat(extended_second_coor, dim=0).long()
                        # Update the dictionary
                        indices_dict[ico] = [extended_first_coor, extended_second_coor]
                else:
                    # Handle edge_indices and downsample_indices
                    for ico, indices in indices_dict.items():
                        shift = torch.max(indices) + 1
                        extended_indices = [indices]
                        for i in range(1, n_batches):
                            extended_indices.append(indices + (shift * i))
                        extended_indices = torch.cat(extended_indices, dim=1) if indice_type == 'edge_indices' else torch.cat(extended_indices, dim=0)
                        # Update the dictionary
                        indices_dict[ico] = extended_indices.long()

            # Update the indices
            self.edge_indices = absolute_indices['edge_indices']
            self.downsample_indices = absolute_indices['downsample_indices']
            self.upsampling_toolkit = absolute_indices['upsampling_toolkit']

            return 

        def forward(self, gnn_data):

            x, batch_size = gnn_data.x, gnn_data.num_graphs
            #start_time = time.time()

            # Check if batch preprocessing needs to be performed
            try: assert (self.last_batch_size == batch_size)
            except AssertionError:
                self.batch_processed = False

            if not self.batch_processed: # defaults to True for batch size 1
                # Extend the indices with respect to the batch size
                self.batch_process(batch_size)
                # Clear the cache for the edge indices
                for layer in self.gcn_layers: layer._cached_edge_index = None
                # Update the batch processing status
                self.batch_processed = True
                # Update the batch sizes
                self.last_batch_size = batch_size

            # Encoding phase

            ### First block ###
            gnn_x_6 = self.gcn1(x, self.edge_indices[self.ico_levels[0]]) # 5 -> 16
            gnn_x_6 = self.bn1(gnn_x_6)
            gnn_x_6 = self.relu1(gnn_x_6)

            ### Second block ###
            gnn_x_5 = self.downsample_block(gnn_x_6, ico=self.ico_levels[0]) # ico 6 -> ico 5
            gnn_x_5 = self.gcn2(gnn_x_5, self.edge_indices[self.ico_levels[1]])  # 16 -> 32
            gnn_x_5 = self.bn2(gnn_x_5)
            gnn_x_5 = self.relu2(gnn_x_5)
            gnn_x_5 = self.dropout1(gnn_x_5)

            # Embedding phase 

            ### Third block ###
            gnn_x_4 = self.downsample_block(gnn_x_5, ico=self.ico_levels[1]) # ico 5 -> ico 4
            gnn_x_4 = self.gcn3(gnn_x_4, self.edge_indices[self.ico_levels[2]]) # 32 -> 32
            gnn_x_4 = self.bn3(gnn_x_4)
            gnn_x_4 = self.relu3(gnn_x_4)
            gnn_x_4 = self.dropout2(gnn_x_4)

            # Decoding phase 
            
            ### Fourth block ###
            gnn_x1_5 = self.upsample_block(gnn_x_4, ico=self.ico_levels[2]) # ico 4 -> ico 5
            gnn_x1_5 = torch.cat((gnn_x1_5, gnn_x_5), dim=1) # skip connection; 32 -> 48
            #
            gnn_x1_5 = self.gcn4(gnn_x1_5, self.edge_indices[ico_levels[1]])  # 48 -> 16
            gnn_x1_5 = self.bn4(gnn_x1_5)
            gnn_x1_5 = self.relu4(gnn_x1_5)

            ### Fifth block ###
            gnn_x1_6 = self.upsample_block(gnn_x1_5, ico=ico_levels[1]) # ico 5 -> ico 6
            gnn_x1_6 = torch.cat((gnn_x1_6, gnn_x_6), dim=1) # skip connection; 16 -> 32
            #
            gnn_x1_6 = self.gcn5(gnn_x1_6, self.edge_indices[ico_levels[0]]) # 32 -> 16
            gnn_x1_6 = self.bn5(gnn_x1_6)
            gnn_x1_6 = self.relu5(gnn_x1_6)

            ### Sixth block ###
            gnn_x1_6 = self.gcn6(gnn_x1_6, self.edge_indices[ico_levels[0]]) # 16 -> 1

            #end_time = time.time()
            #print(f"Elapsed time: {end_time - start_time} seconds")
            return gnn_x1_6.squeeze()
        
    built_nn = gnn_model()
    return built_nn

# Class for training and testing a neural network
class nn_builder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model # untrained model object
    
    def train_nn(self, X_train, y_train, batch_size, 
                batch_load, n_epochs, lr, print_every, criterion, weight_decay=0.01):
        '''
        train the neural network
            assumes identical graph for all subjects
        return: trained model
        '''

        # Set the seed
        set_seed(SEED)

        # Send the model to cuda
        model = self.model.to(device)

        # Prepare to train the model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()

        # Load the data as a PYG object
        data_list = [None] * X_train.shape[0]
        for i in range(X_train.shape[0]):
            data_list[i] = Data_pyg(x=X_train[i], y=y_train[i])
        loader_gnn = DataLoader_pyg(data_list, batch_size=batch_load, shuffle=False) # False for determinism

        # Determine how many batches to run per weight update
        n_batch_iterations = batch_size // batch_load

        # Create variables to save the best loss and best model
        best_loss = float('inf')

        for i in range(n_epochs):
            epoch_loss = 0.0
            total_samples = 0

            for idx, batch in enumerate(loader_gnn):

                # Send the batch to cuda
                batch = batch.to(device)

                # Expand the age to fill the input/output shape
                batch.y = torch.repeat_interleave(batch.y, repeats=self.n_vertices) # number per batch, repeated per n_vertices

                # Get the model loss
                output = model(batch)
                if criterion == 'variance_and_mae':
                    loss = self.variance_and_mae(output, batch.y)
                else:
                    loss = criterion(output, batch.y)

                loss.backward() # accumulate gradients

                # Accumulate loss and count samples
                epoch_loss += loss.item() * batch.num_graphs
                total_samples += batch.num_graphs

                # Complete the loss calculation if the batch size is reached, then reset the gradients
                if (idx + 1) % n_batch_iterations == 0:
                    optimizer.step() # update weights
                    optimizer.zero_grad() # clear gradients

            # Handle the final batch seperately if it hasn't been used to update the gradients
            if (idx + 1) % n_batch_iterations != 0:
                optimizer.step()
                optimizer.zero_grad()

            # Get average epoch loss
            avg_epoch_loss = epoch_loss / total_samples

            # Print every x epochs
            if ((i + 1) % print_every) == 0:
                if criterion == F.l1_loss:
                    print(f'Epoch: {i + 1}, MAE: {avg_epoch_loss:.3f}')
                else:
                    print(f'Epoch: {i + 1}, "{criterion}" Loss: {avg_epoch_loss:.3f}')

            # Save the best model params
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_params = deepcopy(model.state_dict())

        # Return the trained model
        model.load_state_dict(best_model_params)
        return model

    @torch.no_grad()
    def test_nn(self, X_test, y_test, batch_load):
        '''
        Test the neural network
        '''

        # Grab the model
        model = self.model.to(device)
        model.eval()

        # Load the data as a PYG object
        data_list = [None] * X_test.shape[0]
        for i in range(X_test.shape[0]):
            data_list[i] = Data_pyg(x=X_test[i], y=y_test[i])
        loader_gnn = DataLoader_pyg(data_list, batch_size=batch_load, shuffle=False) # False for determinism

        # Pre-emptively put y_test on the GPU for accuracy computations
        y_test = y_test.to(device)
        
        # Initialize accumulators
        brain_ages = torch.zeros(y_test.shape[0], device=device)  # global (avg) predictions
        sum_mae = torch.tensor(0.0, device=device)  # sum of MAE for all batches
        sum_vae = torch.tensor(0.0, device=device)  # sum of VAE for all batches
        sum_per_node_error = torch.zeros(self.n_vertices, device=device)  # raw error per vertex
        pred_per_vertex = torch.zeros((y_test.shape[0], self.n_vertices), device=device) # all predictions per vertex
        total_samples = 0  # counter for total processed samples

        for batch_idx, batch in enumerate(loader_gnn):
            # Send the batch to GPU
            batch = batch.to(device)

            # Expand the age to fill the input/output shape
            batch.y = torch.repeat_interleave(batch.y, repeats=self.n_vertices)

            # Get the model output
            output = model(batch)

            # Save this for the working subjects
            start_idx = batch_idx * batch_load
            end_idx = start_idx + batch.num_graphs
            pred_per_vertex[start_idx:end_idx] = output.view(batch.num_graphs, self.n_vertices)

            # Get the error per node
            error = output - batch.y  # BA - CA

            # Get the whole-brain (average) prediction
            whole_brain_pred = output.view(batch.num_graphs, self.n_vertices).mean(dim=1)
            start_idx = batch_idx * batch_load
            end_idx = start_idx + batch.num_graphs
            brain_ages[start_idx:end_idx] = whole_brain_pred

            # Accumulate loss
            index = torch.arange(error.shape[0], device=device) % self.n_vertices
            sum_per_node_error.scatter_add_(0, index, error)
            
            # Calculate and accumulate losses weighted by batch size
            sum_mae += error.abs().mean() * batch.num_graphs
            sum_vae += self.variance_and_mae(output, batch.y) * batch.num_graphs
            total_samples += batch.num_graphs

        # Compute averages using actual processed sample count
        # note that this may be slightly inaccurate since its an avg of averages, 
        # and there may be a single batch that's smaller. But this is negligible
        avg_mae = sum_mae / total_samples
        avg_vae = sum_vae / total_samples
        per_node_e = sum_per_node_error / total_samples

        print(f'MAE (L1) Loss: {avg_mae.item():.3f} across {total_samples} observations')
        print(f'Variance and MAE Loss: {avg_vae.item():.3f} across {total_samples} observations')

        # Get the whole-brain error
        age_gaps = brain_ages - y_test #(brain_ages - y_test)

        return avg_mae, per_node_e, y_test, age_gaps, pred_per_vertex

    def single_cv(self, X, y, k_folds, batch_size, batch_load, 
                    n_epochs, lr, print_every, criterion, weight_decay): # change later to False?
        '''
        Perform k-fold cross validation
        '''
        
        # Store initial model state
        init_state = self.model.state_dict()
        n_samples = y.shape[0]

        # Pre-allocate tensors for accumulation
        fold_losses = torch.zeros(k_folds, device=device)  # whole-brain MAE, per fold
        fold_errors = torch.zeros((k_folds, self.n_vertices), device=device)  # per-vertex error, per fold
        all_chr_ages = torch.zeros(n_samples, device=device)  # chronological ages (actual brain ages)
        all_age_gaps = torch.zeros(n_samples, device=device)  # normalized whole-brain errors
        all_pred_per_vertex = torch.zeros((n_samples, self.n_vertices), device=device) # predictions for each participant, for each vertex
        fold_sample_counts = torch.zeros(k_folds, dtype=torch.long, device=device)  # track samples per fold

        # Create KFold splitter
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=SEED) # Seed for determinism

        # Determine where to save fold outputs too
        # TODO implement this as a parameter that is propogated through the Class
        fold_output_path = '/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/last_model_outputs/fold_values/'

        for fold, (train_idx, test_idx) in enumerate(kf.split(range(n_samples))):
            print(f'\n=== Fold {fold + 1}/{k_folds} ===')
            print(f'Train samples: {len(train_idx)}, Test samples: {len(test_idx)}')

            # Reset model by loading initial state
            self.model.load_state_dict(init_state)
            self.model.to(device)

            # If a failed iteration already completed this, save the files
            if os.path.exists(f'/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/last_model_outputs/fold_values/avg_mae_fold_{fold}.pt'):
                
                # These automatically will retain the saved device
                avg_mae = torch.load(f'{fold_output_path}avg_mae_fold_{fold}.pt')
                per_node_e = torch.load(f'{fold_output_path}per_node_e_fold_{fold}.pt')
                chr_ages = torch.load(f'{fold_output_path}chr_ages_fold_{fold}.pt')
                age_gaps = torch.load(f'{fold_output_path}age_gaps_fold_{fold}.pt')
                pred_per_vertex = torch.load(f'{fold_output_path}pred_per_vertex_fold_{fold}.pt')

                print(f'MAE (L1) Loss: {avg_mae.item():.3f} across {len(test_idx)} observations')

            # If there is no saved file for the previous iterations 
            else:

                # Train the model
                self.model = self.train_nn(
                    X[train_idx], y[train_idx], 
                    batch_size=batch_size, 
                    batch_load=batch_load, 
                    n_epochs=n_epochs, 
                    lr=lr, 
                    print_every=print_every, 
                    criterion=criterion,
                    weight_decay=weight_decay
                )

                # Evaluate the model
                avg_mae, per_node_e, chr_ages, age_gaps, pred_per_vertex = self.test_nn(
                    X[test_idx], y[test_idx], 
                    batch_load=batch_load
                )

                # Save results
                torch.save(avg_mae, f'{fold_output_path}avg_mae_fold_{fold}.pt')
                torch.save(per_node_e, f'{fold_output_path}per_node_e_fold_{fold}.pt')
                torch.save(chr_ages, f'{fold_output_path}chr_ages_fold_{fold}.pt')
                torch.save(age_gaps, f'{fold_output_path}age_gaps_fold_{fold}.pt')
                torch.save(pred_per_vertex, f'{fold_output_path}pred_per_vertex_fold_{fold}.pt')

            # Store results
            start_idx = fold_sample_counts[:fold].sum().item()
            end_idx = start_idx + len(test_idx)

            fold_losses[fold] = avg_mae  # average whole-brain MAE
            fold_errors[fold] = per_node_e  # average per-vertex error
            all_chr_ages[start_idx:end_idx] = chr_ages  # whole-brain age (actual; chronological age)
            all_age_gaps[start_idx:end_idx] = age_gaps  # normalized whole-brain errors
            all_pred_per_vertex[start_idx:end_idx] = pred_per_vertex # predictions for each participant, for each vertex
            fold_sample_counts[fold] = len(test_idx) # update number of completed subjects

            # Clean up
            torch.cuda.empty_cache()

        # Compute weighted averages across folds
        weights = fold_sample_counts.float() / n_samples # weigh the folds based on the number of samples
        avg_loss = (fold_losses * weights).sum()  # average loss across all folds
        avg_error = (fold_errors * weights.view(-1, 1)).sum(dim=0)  # average error across all folds on a per-vertex level
        
        print(f'\n=== Final Results ===')
        print(f'Average MAE across {k_folds} folds: {avg_loss.item():.3f}')
        print(f'Sample distribution per fold: {fold_sample_counts.tolist()}')

        torch.save(avg_loss, f'{fold_output_path}avg_loss.pt')
        torch.save(avg_error, f'{fold_output_path}avg_error.pt')
        torch.save(all_chr_ages, f'{fold_output_path}all_chr_ages.pt')
        torch.save(all_age_gaps, f'{fold_output_path}all_age_gaps.pt')
        torch.save(all_pred_per_vertex, f'{fold_output_path}all_pred_per_vertex.pt')

        return avg_loss, avg_error, all_chr_ages, all_age_gaps, all_pred_per_vertex

    def vae_preproc(self, fsavg_path, batch_load, first='rh'):

        # Get the labels for both hemis for both paths
        rh_labels, _, _ = nib.freesurfer.read_annot(f'{fsavg_path}label/rh.aparc.a2009s.annot')
        lh_labels, _, _ = nib.freesurfer.read_annot(f'{fsavg_path}label/lh.aparc.a2009s.annot')

        # Combine these into a single np array
        if first == 'rh': # ico7 has -1; will need to account for that seperately, not done here
            labels = np.hstack((rh_labels, lh_labels+np.max(rh_labels)+1))
        else: 
            labels = np.hstack((lh_labels, rh_labels+np.max(lh_labels)+1))

        # Tensor conversion and preprocessing
        labels = torch.tensor(labels, dtype=torch.long, device=device) - 1 # so 0-based ; may cause issues for ico-7 since it has -1
        
        # Batch-aware label expansion
        shift = torch.max(labels)+1
        batch_labels = torch.cat(
            [labels + (shift * i) for i in range(batch_load)],
            dim=0
        )

        # Precompute indices
        # inverse_indices removes gaps between indices (e.g. 0, 1, 3 -> 0, 1, 2)
        _, inverse_indices, counts = torch.unique(batch_labels, return_inverse=True, return_counts=True)    
        self.inverse_indices = inverse_indices.to(device)   
        self.counts_float = counts.float().to(device)
        
        return
    
    def variance_and_mae(self, y_pred, y_actual, mean=True, scale=2, pairwise=True): 

        # === Compute variance per-region, per-subject === 

        # Prune the labels to account for smaller batch sizes
        pruned_labels = self.inverse_indices[:y_pred.shape[0]] # relate each vertex to a group, unique to each region/subject

        # Compute region means using scatter_mean
        region_means = scatter_mean(y_pred, pruned_labels, dim=0) # = μ ;; by group

        # Subtract the mean from each prediction 
        squared_diff = (y_pred - region_means[pruned_labels])**2 # = (x - μ)^2 ;; by vertex

        # Sum the squared differences
        summed_diff = torch.zeros_like((region_means))
        summed_diff.scatter_add_(0, pruned_labels, squared_diff) # = ∑ (x - μ)^2 ;; by group

        # Divide each prediction by the number of vertices in the region
        region_var = summed_diff / self.counts_float[:summed_diff.shape[0]] # = σ^2 ;; by group

        # Copy this value for every vertice within its associated region
        label_variances = region_var[pruned_labels] # = σ^2 ;; by group, repeated by vertex

        # === Compute global variance, per-subject === #
        
        if pairwise: # if computing global seperation based off of pairwise distance between category means

            # Get the variance across the region means, maintaining batch separation
            subj_avg_dist = torch.zeros(self.model.last_batch_size, device=device) # shape (n_subjects,)
            n_labels_per = len(region_means) // self.model.last_batch_size

            for idx, batch_start in enumerate(range(0, len(region_means), n_labels_per)):
                batch_end = batch_start + n_labels_per
                current_batch = region_means[batch_start:batch_end]

                # Find the euclidean difference between every label mean
                pairwise_d = torch.cdist(current_batch.view(-1, 1), current_batch.view(-1, 1), p=1) # shape (n_labels, n_labels-1)

                # Get the average pairwise distance per-label
                subj_avg_dist[idx] = (torch.sum(pairwise_d) / (n_labels_per * (n_labels_per-1))) # don't need 2x in the numerator because each combination is present twice

            # Create per-vertex output by repeating each subject's avg pairwise distances
            global_per_vertex = subj_avg_dist.repeat_interleave(self.n_vertices)

        else: # if computing global variance based on all vertices

            # Get the subjects as seperate rows
            subjects = y_pred.view(-1, self.n_vertices)

            # Get the variance per-subject
            subject_variances = torch.var(subjects, dim=1) # shape (n_subjects,)

            # Create per-vertex output by repeating each subject's variance
            global_per_vertex = subject_variances.repeat_interleave(self.n_vertices)

        # === Compute cumulative loss, per-vertex, per-subject === #
        ae_loss = torch.abs(y_pred - y_actual)
        ae_loss = ae_loss ** scale # 1 = MAE, 2 = MSE, etc.

        total_loss = ae_loss - (global_per_vertex * self.global_w) + (label_variances * self.intra_w)
        return total_loss.mean() if mean else total_loss

    def __call__(self, X_train, y_train, X_test, y_test,
                 model=None, batch_size=32, batch_load=1, 
                 n_epochs=50, lr=0.0001, print_every=10, 
                 ico_levels=[6, 5, 4], criterion=F.l1_loss, 
                 weight_decay=0.01, first='rh',
                 intra_w=0.1, global_w=0.05):
        '''
        Manipulate the NN
        '''

        # Determine the number of vertices based on the top ico level
        ico_dict={4:5124, 5:20484, 6:81924, 7:327684} # across both hemispheres
        self.n_vertices = ico_dict[ico_levels[0]]

        # Preprocess for variance_and_mae (noted in test even if not used to train)
        # Get the fsavg path
        if ico_levels[0] == 7: fsavg_path = f'/mnt/md0/softwares/freesurfer/subjects/fsaverage/' # first in list should be largest
        else: fsavg_path = f'/mnt/md0/softwares/freesurfer/subjects/fsaverage{ico_levels[0]}/'
        # Save relevant indices
        self.vae_preproc(fsavg_path, batch_load, first)
        # Store weights
        self.intra_w = intra_w
        self.global_w = global_w

        # Set seed for reproducability
        set_seed(SEED)

        # If using a trained model
        if model is not None: self.model = model

        # If testing a trained model
        if X_train == None:
            # Save the model and load in the data
            X_test = torch.from_numpy(np.load(X_test).astype(np.float32)); y_test = torch.from_numpy(np.load(y_test).astype(np.float32))
            # Test the model
            avg_mae, per_node_e, chr_ages, age_gaps, pred_per_vertex = self.test_nn(X_test, y_test, batch_load=batch_load)
            torch.cuda.empty_cache(); return avg_mae.cpu().numpy(), per_node_e.cpu().numpy(), chr_ages.cpu().numpy(), age_gaps.cpu().numpy(), pred_per_vertex.cpu().numpy()
                    
        # If training a model with no test set
        elif y_test == None:
            # Load in the data
            X_train = torch.from_numpy(np.load(X_train).astype(np.float32)); y_train = torch.from_numpy(np.load(y_train).astype(np.float32))
            # Train and return the model
            self.model = self.train_nn(X_train, y_train, batch_size=batch_size, 
                                            batch_load=batch_load, n_epochs=n_epochs,
                                            lr=lr, print_every=print_every, criterion=criterion, 
                                            weight_decay=weight_decay)
            torch.cuda.empty_cache(); return self.model
        
        # If performing cross validation
        elif X_train == X_test:
            # Load in the data
            X_train = torch.from_numpy(np.load(X_train).astype(np.float32)); y_train = torch.from_numpy(np.load(y_train).astype(np.float32))
            # Perform 5-fold cross validation
            avg_mae, per_node_e, chr_ages, age_gaps, pred_per_vertex = self.single_cv(X=X_train, y=y_train, k_folds=5,
                                            batch_size=batch_size, batch_load=batch_load, 
                                            n_epochs=n_epochs, lr=lr, 
                                            print_every=print_every, criterion=criterion, weight_decay=weight_decay)
            torch.cuda.empty_cache(); return avg_mae.cpu().numpy(), per_node_e.cpu().numpy(), chr_ages.cpu().numpy(), age_gaps.cpu().numpy(), pred_per_vertex.cpu().numpy()

        # If training then testing a model
        else:
            # Load in the data
            X_train = torch.from_numpy(np.load(X_train).astype(np.float32)); y_train = torch.from_numpy(np.load(y_train).astype(np.float32))
            X_test = torch.from_numpy(np.load(X_test).astype(np.float32)); y_test = torch.from_numpy(np.load(y_test).astype(np.float32))
            # Train and test the model
            self.model = self.train_nn(X_train, y_train, batch_size=batch_size, 
                                            batch_load=batch_load, n_epochs=n_epochs,
                                            lr=lr, print_every=print_every, criterion=criterion,
                                            weight_decay=weight_decay)
            avg_mae, per_node_e, chr_ages, age_gaps, pred_per_vertex = self.test_nn(X_test, y_test, batch_load=batch_load)
            torch.cuda.empty_cache(); return avg_mae.cpu().numpy(), per_node_e.cpu().numpy(), chr_ages.cpu().numpy(), age_gaps.cpu().numpy(), pred_per_vertex.cpu().numpy()