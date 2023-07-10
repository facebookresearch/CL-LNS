# Setting up the environment
This only needs to be done once


## CUDA 

You could use CUDA for training.


## Create the environment 

First install conda (if you haven't) and activate conda.
You also need to decide a torch version to use depending on your cuda version. 
In this repo, we provide an example for torch 1.7.0 with cuda 11.0.
If you are using other torch/cuda versions, modify `environment.txt`, `requirements.txt`  and the commands below accordingly. The code has been tested with torch 1.7.0 and 1.12.1.

After that, you could run the commands below to install the conda environment:
```
conda config --add channels conda-forge
conda config --add channels pytorch
conda config --add channels nvidia
conda config --add channels dglteam
conda create -n cllns --file environment.txt
conda activate cllns
pip cache purge
pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
```

We have an updated version of the Ecole library that fixes some issues with feature extraction and could be downloaded from [here](https://drive.google.com/file/d/1EbnUlgdnJotCKAyh8M9hqsMvaIvIP0Sq/view?usp=share_link).

## Install Ecole
Download Ecole and apply a patch to it. Then, install it from the source:
```
mkdir ecole_0.8.1
cd ecole_0.8.1
git clone -b v0.8.1 https://github.com/ds4dm/ecole.git
patch -p1 < ../ecole_feature_extraction.patch
cd ecole
python -m pip install .
cd ../..
```

# Instances Preparation

In ```instance_loader.py```, add to ```LOCAL_INSTANCE``` the problem set name of the instances and the path to those instances.
In this repo, we provide a mini example for training (`INDSET_train`) and testing (`INDSET_test`) instances for the independset problem used in our paper.



# Collecting the dataset

Run data collection on the example training problem set:
```
python LNS.py --problem-set=INDSET_train  --num-solve-steps=30 --neighborhood-size=100 --time-limit=300000 --destroy-heuristic=LOCAL_BRANCHING --mode=COLLECT
```
Here we are collecting data with the expert Local Branching for 30 iterations with neighborhood size 100. The per-iteration runtime limit is hard-coded in the code for 1 hour (can be changed there). So the total time limit needs to be set to be at least 30*3600 seconds if you want to finish all 30 iterations.



## Train the models


Install some packages for contrastive losses and keeping track of other metrics:
```
pip install pytorch-metric-learning
pip install torchmetrics==0.9.3
```


After that, simply run 

```python LNS/train_neural_LNS.py```

## Test the models


Run this command to test the model on the example testing problem set. This command use an example model for testing that is also used in our experiment.
```
python LNS.py --problem-set=INDSET_test --neighborhood-size=3000 --time-limit=3600 --destroy-heuristic=ML::GREEDY::CL --adaptive=1.02 --mode=TEST_ML_feat2 --gnn-type=gat --model=model/INDSET_example.pt
```
ML-base destroy heuristics must have prefix `ML::SAMPLE` or `ML::GREEDY` (depending on the sampling methods you use, for CL-LNS we recommend GREEDY) and can have arbitrary suffix (here we use `::CL`) for your own naming and logging purposes. 

# Citation

Please cite our paper if you use this code in your work.
```
@inproceedings{huang2023searching,
  title={Searching Large Neighborhoods for Integer Linear Programs with Contrastive Learning},
  author={Huang, Taoan and Ferber, Aaron and Tian, Yuandong and Dilkina, Bistra and Steiner, Benoit},
  booktitle={International conference on machine learning},
  year={2023},
  organization={PMLR}
}
```

# LICENSE
The project is under CC BY-NC 4.0 license. Please check LICENSE file for details.
