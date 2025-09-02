import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

target_icos = [3, 4, 5, 6, 7]

all_vertices = {}
all_edges = {}
for ico in target_icos:

    # get the relevant spatial path
    if ico == 7: fsavg_path = '/mnt/md0/softwares/freesurfer/subjects/fsaverage/'
    else: fsavg_path = f'/mnt/md0/softwares/freesurfer/subjects/fsaverage{ico}/'

    # get the vertice coordinates for the target ico, based on the spherical coordinates
    vertices, faces = nib.freesurfer.read_geometry(f"{fsavg_path}surf/rh.sphere")
    all_vertices[ico] = vertices

    # get the edges from the faces
    edges = np.vstack([
        faces[:, [0, 1]],  # edge 1: v1, v2
        faces[:, [1, 2]],  # edge 2: v2, v3
        faces[:, [2, 0]],   # edge 3: v3, v1
        faces[:, [1, 0]],  # reverse of edge 1
        faces[:, [2, 1]],  # reverse of edge 2
        faces[:, [0, 2]]   # reverse of edge 3
    ])
    # sort the edges, remove duplicates, and transpose
    sorted_edges = np.sort(edges, axis=1)
    unique_edges = np.unique(sorted_edges, axis=0)
    all_edges[ico] = unique_edges.T

all_vertices = {}
all_edges = {}

for ico in target_icos:

    # get the relevant spatial path
    if ico == 7: fsavg_path = '/mnt/md0/softwares/freesurfer/subjects/fsaverage/'
    else: fsavg_path = f'/mnt/md0/softwares/freesurfer/subjects/fsaverage{ico}/'

    # get the vertice coordinates for the target ico, based on the spherical coordinates
    vertices, faces = nib.freesurfer.read_geometry(f"{fsavg_path}surf/rh.sphere")
    all_vertices[ico] = vertices

    # get the edges from the faces
    edges = np.vstack([
        faces[:, [0, 1]],  # edge 1: v1, v2
        faces[:, [1, 2]],  # edge 2: v2, v3
        faces[:, [2, 0]],   # edge 3: v3, v1
        faces[:, [1, 0]],  # reverse of edge 1
        faces[:, [2, 1]],  # reverse of edge 2
        faces[:, [0, 2]]   # reverse of edge 3
    ])
    # sort the edges, remove duplicates, and transpose
    sorted_edges = np.sort(edges, axis=1)
    unique_edges = np.unique(sorted_edges, axis=0)
    all_edges[ico] = unique_edges.T

root_dir = '/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/pooling/'

# iterate through the icos and get the downsampled and upsampled indices
for ico in all_vertices.keys():
        
    # check if the ico is the top or bottom
    if ico+1 in all_vertices.keys():
        '''
        upsample process:
        get the 1 nearest neighbor in the higher ico, then get its receptive field based on its edge index
        '''
        
        # create a tree using the ico to upsample into
        k=1
        tree = cKDTree(all_vertices[ico+1])

        # get the equivalent index within the upsampled ico, from the downsampled ico
        _, indices = tree.query(all_vertices[ico], k=k)

        # get the neighbors for each of these indic
        receptive_field = np.zeros((indices.shape[0], 7)) # equivalent and neighborhood
        for index in indices:
            # only need to do this in one direction since edge index is bidirectional
            matching_indices = np.where(all_edges[ico+1][0] == index)
            corresponding_values = all_edges[ico+1][1, matching_indices][0]
            # include the source node
            corresponding_values = np.concatenate((corresponding_values, [index]))
            # if the node has 5 neighbors; should be 12 of these
            if corresponding_values.shape == (6,):
                corresponding_values = np.concatenate((corresponding_values, [index]))
            receptive_field[index, :] = corresponding_values

        # make the indices into integers
        receptive_field = receptive_field.astype(int)

        # stack this once more to account for both hemispheres
        combined_indices = np.vstack((receptive_field, receptive_field+np.max(receptive_field)+1))
        print(f'ico: {ico}, shape: {combined_indices.shape}, max: {np.max(combined_indices)}, path name: {root_dir}ico{ico+1}->ico{ico}_downsample.npy')

        # save these
        np.save(f'{root_dir}ico{ico+1}->ico{ico}_downsample.npy', combined_indices)

