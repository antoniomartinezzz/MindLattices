# MindLattices

Current brain graph construction methods for functional connectivity studies involve using a priori atlases to partition the brain into regions of interest. These methods often result in information loss and fail to capture functional variability across individuals. To address these limitations, we propose to partition the brain into overlapping regions for each individual subject, using a three-dimensional sliding window. To evaluate the performance of our parcellation technique, we train and evaluate **BrainGNN** and **Brain Network Transformer** (BNT) on a classification task using the Autism Brain Imaging Data Exchange (ABIDE) I dataset and compare the model’s performance with two commonly used a-priori ROI atlases, Craddock 200 (**CC200**) and Harvard-Oxford (**HO**). The results suggest that our parcellation strategy achieves comparable performance compared to conventional methods. The models present unstable performance due to noise in the dataset which makes experimentation and optimization difficult. Performance variability across dataset distributions does not allow for a fair comparison across parcellation strategies and across studies. We highlight the need of a well defined benchmark to facilitate the study of this problem. 


We evaluate and compare the utility of the baseline parcellation strategies (CC200 and HO) with our parcellation strategy using BrainGNN and BNT on the ABIDE I dataset, using a 5-fold cross-validation with five repetitions. 

## BrainGNN
See the BrainGNN.yaml for environment configuration.
### Installation
Clone the repository 

Run these lines to create and activate an appropriate 
environment:

```conda create --name yourEnv python=3.8```

```conda activate yourEnv```

```pip install nilearn```

```pip install deepdish```

```pip install tensorboardX==2.2```

```pip install protobuf==3.17.0```

```pip install torch-geometric==1.7.0```

```conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch```

```pip3 install torch-sparse==0.6.9 --no-cache-dir -f https://data.pyg.org/whl/torch-1.7.0+cu110.html```

```pip3 install torch-scatter==2.0.6 --no-cache-dir -f https://data.pyg.org/whl/torch-1.7.0+cu110.html```

```pip install networkx==2.5```

```pip install numpy==1.23.4```

### Test

To test the pretrained BrainGNN model using the 5-fold cross-validation with five repetitions. 


### Train-test

### Demo



## BNT

### Installation

### Test

To test the pretrained BrainGNN model 

### Train-test

### Demo
