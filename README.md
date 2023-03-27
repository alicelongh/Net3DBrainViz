# network-visualization
scripts I created/used for visualisation of networks

1) 3D viz of arrays of values using visbrain:

- using BCI-NET fork of visbrain: install https://github.com/BCI-NET/visbrain with > pip install git+https://github.com/BCI-NET/visbrain.git

- using mni152 brain template: install template brain data  https://drive.google.com/file/d/1vCSQC7csXBY8eOBOEEhXqXqcp5qVj4jT/view
*I don’t remember where it should be installed. Initially I had it in /Users/alice.longhena/visbrain_data/
  
- insert the path to NIFTI files at the scale you are working on, NIFTI files generated with easy_lausanne https://github.com/mattcieslak/easy_lausanne.git (Lausanne2008 parcellation) and free surfer

- modify 3DbrainViz-Lausanne2008.py with the paths to the input array and output path and file
	
- the script creates (1) a panel of the cortex view and (2) another panel showing the interior resulting from saggital plane slice
*it would be nice to add the other planes, horizontal and coronal

![example cortex view](https://user-images.githubusercontent.com/57717790/227973018-c07b3394-fca9-4693-a985-e5ab7bf91a03.png)
![example saggital plane slice view](https://user-images.githubusercontent.com/57717790/227973112-b661c100-5d0a-4537-adaf-c1ea989809c3.png)
