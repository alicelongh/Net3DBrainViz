#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:37:04 2023

@author: alice.longhena

- Takes a certain parcellation (here Lausanne 2008) of the brain in ROIs, equivalent to network nodes

- Takes values computed for network nodes

- Takes a template brain

------------------------

- Represents those values on the corresponding brain regions on a template 3D brain

- Outputs: panel of the cortex from different prospectives, panel of the internal regionrs
        if threshold + True outputs also a file containing the regions with p-val smaller than threshold
        
"""

from visbrain.objects import RoiObj, BrainObj, SceneObj, ColorbarObj
from visbrain.gui import brain
from visbrain.utils import volume_to_mesh, volume_to_data
from visbrain.io import download_file
from visbrain.utils.mesh import mesh_edges
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
from copy import deepcopy
import matplotlib.pyplot as plt
from visbrain.gui import Figure
from time import time
import os
import scipy
import scipy.io
from scipy.sparse import coo_matrix


def coregister_mesh_to_vol(vert, vol):

    vox_xyz = np.argwhere(vol)

    vert_xyz_size = np.ptp(vert, axis=0)
    vol_xyz_size = np.ptp(vox_xyz, axis=0)

    scale_factor = vol_xyz_size / vol_xyz_size
    vert = vert * scale_factor

    vert_cent = np.mean(vert, axis=0)
    vol_cent = np.mean(vox_xyz, axis=0)

    trans_factor = vol_cent - vert_cent
    vert += trans_factor

    vert_xyz_size = np.ptp(vert, axis=0)

    vert_cent = np.mean(vert, axis=0)
    print(vert_cent, vol_cent)


    # print(vert_xyz_size, vol_xyz_size)

    return vert



#%% cell VOLUME CREATION

## INIT -----
folder_proj = "/Users/alice.longhena/ownCloud/Alice2223/PROJECTS/3DbrainViz"

# scale:number of nodes --> 33:82 , 60:128 , 125:233 , 250:462 , 500:1014(missing)
scale = 60
num_nodes = 128

# path to nii.gz file
p = "%s/Lausanne_NIFTI/icbm152_reference-brain-VOLUMES/easy_lausanne_output/scale%i/ROIv_HR_th.nii.gz" %(folder_proj,scale)

#------------

CBAR_STATE = dict(cbtxtsz=12, txtsz=10., width=.10, cbtxtsh=3.,
                  rect=(-.3, -2., 1., 4.))

KW = dict(title_size=14., zoom=1000)

start = time()

start_nii = time()
print(f"Scene created in {start_nii-start}s")


## open volume file

p_img = nib.load(p)
p_vol = p_img.get_fdata()
#p_hdr = p_img.affine

vol = p_vol
roi_labels = np.unique(vol[np.where(vol != 0)])

end_nii = time()
print(f"Nifti opened in {end_nii-start_nii}s")



#%% cell VOLUME MANPULATION 

## INIT -----

# TITLE
measure_string = "gnd score" # GOES IN TITLE, name of the observable we viz


# OUTPUT path
path_out = "%s/e191919N128/" %(folder_proj) #folder in which images go
if os.path.exists(path_out)==False:
    os.makedirs(path_out)  
    

# INPUT
path_node_values = "%s"%(path_out)+"p-test-N=128,knn=12,#perm=1000.txt" 
node_values = np.genfromtxt(path_node_values)[:,1] #array to viz


anat_labels = np.genfromtxt("%s/Lausanne_NIFTI/ParcellationLausanne2008_Scale%i.txt" %(folder_proj,scale), dtype='str')[1:,1]

cmap_string = 'Spectral_r'

# ------------

# normalize
node_values = np.abs(node_values-np.average(node_values))/np.std(node_values) 
               

## copy vol and loop on voxels to change labels into node measure values we want to represent                

data_vol = np.copy(vol)

for i in range(num_nodes):
    l = len(data_vol[np.where(vol==i+1)])
    if np.isnan(node_values[i])==True:
        data_vol[np.where(vol==i+1)]=0
    else:
        data_vol[np.where(vol==i+1)]=node_values[i]*10**2
      
                                       
## converting our nifti volume to an array of vertices, faces(triangular area between vertices) and norm
# vert, faces, norm = volume_to_mesh(vol, smooth_factor=3)
# creating our brain object with our custom vert, faces and norm

b_obj = BrainObj('mni152', translucent=False, hemisphere='both') #!!where does the file mni152.npz needs to be?

end_bobj = time()
print(f"BrainObj created in {end_bobj-end_nii}s")


## assign data to vertices of the mesh

vert = coregister_mesh_to_vol(b_obj.vertices, vol)


# For each mesh vertex, we extract the value of the closest voxel if its close enough
# data_vol can be a volume smaller than whole brain
# data_vol=vol if covering the whole brain with data
data, valid_vertices = volume_to_data(data_vol, vert, select=None, radius=3.)

end_data = time()
print(f"Data created in {end_data-end_bobj}s")


# getting the bound for the colormap
data_max = np.round(np.max(data))
data_min = np.round(np.min(data))

#Adding the extracted data to the brain with the colormap 
b_obj.add_activation(data, vertices=valid_vertices, cmap=cmap_string, clim=(data_min,data_max), under='grey', vmin=0)

end_act =  time()
print(f"Activation added in {end_act-end_data}s")


## save mesh info
save_mesh = False

if save_mesh==True:
    
    path_out_save_mesh = '%s/Lausanne_NIFTI/mesh_labels/'%folder_proj
    if os.path.exists(path_out_save_mesh)==False:
        os.makedirs(path_out_save_mesh)
        
    #SAVE MESH info
    amatrix_mesh_sparse = mesh_edges(b_obj.faces)
    #amatrix_mesh = coo_matrix.todense(amatrix_mesh_sparse)[valid_vertices][valid_vertices] #too heavy
    scipy.io.savemat(path_out_save_mesh + "amatrix_mesh_scale%i_N%i.mat" % (scale,num_nodes), {'amatrix_mesh_sparse_coo':amatrix_mesh_sparse})
    #for instance I need mesh with labels for cluster statistics
    #!!! save either both data and binary amatrix_mesh, or amatrix_mesh with labels as entrances
    with open(path_out_save_mesh + "meshxlabels_scale%i_N%i" % (scale,num_nodes), 'w+') as file_mesh:
        np.savetxt(file_mesh, np.column_stack([valid_vertices, data]), header="valid vertices, data on vertices")
 
    
#%% cell: 3D BRAIN PLOT

## plot and save
background_color='white'
text_color='black'

#creating an animation and previewing it (close the preview window because it pauses the code)
#b_obj.animate()
#b_obj.record_animation("test.gif", n_pic=5)
#b_obj.preview()

# taking screenshot to make a panel
b_obj.rotate("front")
b_obj.screenshot(path_out+"front_%s.png" %measure_string, print_size=(5, 5),  bgcolor=background_color, autocrop=True)
b_obj.rotate("side-fl")
b_obj.screenshot(path_out+"fl_%s.png" %measure_string, print_size=(5, 5), bgcolor=background_color, autocrop=True)
b_obj.rotate("left")
b_obj.screenshot(path_out+"left_%s.png" %measure_string, print_size=(5, 5), bgcolor=background_color, autocrop=True)
b_obj.rotate("side-bl")
b_obj.screenshot(path_out+"bl_%s.png" %measure_string, print_size=(5, 5), bgcolor=background_color, autocrop=True)
b_obj.rotate("side-br")
b_obj.screenshot(path_out+"br_%s.png" %measure_string, print_size=(5, 5), bgcolor=background_color, autocrop=True)
b_obj.rotate("right")
b_obj.screenshot(path_out+"right_%s.png" %measure_string, print_size=(5, 5), bgcolor=background_color, autocrop=True)
b_obj.rotate("side-fr")
b_obj.screenshot(path_out+"fr_%s.png" %measure_string, print_size=(5, 5), bgcolor=background_color, autocrop=True)
b_obj.rotate("top")
b_obj.screenshot(path_out+"top_%s.png" %measure_string, print_size=(5, 5), bgcolor=background_color, autocrop=True)

end_sc = time()
print(f"Activation added in {end_act-end_data}s")
print(f"Screenshot saved in {end_sc-end_act}s")

#making the panel
#files = [path_out+"br_%s.png" %measure_string, path_out+"right_%s.png" %measure_string, path_out+"fr_%s.png" %measure_string, path_out+"top_%s.png" %measure_string, path_out+"fl_%s.png" %measure_string, path_out+"left_%s.png" %measure_string, path_out+"bl_%s.png" %measure_string]
files = ["br_%s.png" %measure_string,"right_%s.png" %measure_string, "fr_%s.png" %measure_string, "top_%s.png" %measure_string, 
         "fl_%s.png" %measure_string, "left_%s.png" %measure_string, "bl_%s.png" %measure_string]
titles = ['Back-right', 'Right', 'Front-right', 'Top', 'Front-left', 'Left', 'Back-left']

f = Figure(files, path=path_out, titles=titles, figtitle='%s panel - %i ROIs' %(measure_string, num_nodes),
           grid=(1, 7), y=1., fig_bgcolor=background_color, figsize=(20, 6),
           text_color=text_color, autocrop=True) # ! this function often produces an error

f.shared_colorbar(cmap=cmap_string, clim=(data_min/10**2,data_max/10**2), fz_title=15, position='bottom',
                  title='dev from mean', fz_ticks=10, figmargin=0.1, ycb=5, height=0.52)


# Save the picture :
f.save(path_out+'panel %s.png' %measure_string, dpi=600)

end_fig = time()
print(f"Figure created and saved in {end_fig-end_sc}s")
# Finally, display the figure :
f.show()


## making the panel with interior view

b_obj.hemisphere = "left"
b_obj.rotate("left")
b_obj.screenshot(path_out+"lh_out %s.png" %measure_string, print_size=(5, 5), bgcolor=background_color)
b_obj.rotate("right")
b_obj.screenshot(path_out+"lh_in %s.png" %measure_string, print_size=(5, 5), bgcolor=background_color)

b_obj.hemisphere = "right"
b_obj.rotate("right")
b_obj.screenshot(path_out+"rh_out %s.png" %measure_string, print_size=(5, 5), bgcolor=background_color)
b_obj.rotate("left")
b_obj.screenshot(path_out+"rh_in %s.png" %measure_string, print_size=(5, 5), bgcolor=background_color)

files = [path_out+"lh_out %s.png" %measure_string, path_out+"lh_in %s.png" %measure_string, path_out+"rh_out %s.png" %measure_string, path_out+"rh_in %s.png" %measure_string]
titles = ['left-out', 'left-in', 'right-out', 'right-in']

f = Figure(files, titles=titles, figtitle='%s panel (ICBM152) - %i ROIs' %(measure_string, num_nodes),
           grid=(2, 2), y=1., fig_bgcolor=background_color, figsize=(10, 10),
           text_color=text_color, autocrop=True)

f.shared_colorbar(cmap=cmap_string, clim=(data_min/10**2,data_max/10**2), fz_title=15, position='bottom',
                  title='dev from mean', fz_ticks=10, figmargin=0.1, ycb=5, height=0.52)


# Save the picture :
f.save(path_out+'panel %s in-out.png' %measure_string, dpi=600)

# Finally, display the figure :
f.show()

