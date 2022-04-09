# DeepDynaTree

This is a pytorch implementation of our research:  
Phylogenetic-informed graph deep learning to classify dynamic transmission clusters in infectious disease epidemics.  
If you find our work useful in your research or publication, please cite our work.

## Dependencies
Please check the dependencies.txt.

## Datasets

- Agent-based simulation model [nosoi] to generate simulated trees and input datatables in R.
- Source code used to simulate the data (including seeds) used in this study, as well as code used in calculating tree shape statistics and transforming annotated
trees for DeepDynaTree classification, are provided.

## Usage examples
 - Preparing dataset  
Both [original and preprocessed datasets] are avaliable. Please uzip them in the root folder. Jupyter notebook files in `aly` provide a detailed preprocessing step by step for both node feature and edge feature. The provided preprocessed dataset is divided into train/validation/test sets.  
Before running the code, put the preprocessed dataset on a desired directory. By default, the data root is set as `data/preprocessed_data/split_rs123`.  
See: [dl/config.py]
```sh
data.add_argument("--ds_name", type=str, default="preprocessed_data", help="The name of dataset")
data.add_argument("--ds_dir", type=str, default="../data/", help="The base folder for data")
data.add_argument("--ds_split", type=str, default="split_rs123")
```

 - Run container  
The container is build by [singularity] and is avaliable at [here]. To run a shell within the container:
```sh
singularity shell --nv singularity_deepdynatree.sif    
```

 - Train models  
The main function is located in `dl`, which supports training GCN, GAT, GIN, and PDGLSTM models. Also, it supports training node-based neural network models like MLP, SetTransformer, DeepSet and TabNet.  
Other node-based methods (i.e. LR, RF, XGBoost) are included in [models/ml_models.ipynb].  
Example commands 
```sh
# train a PDGLSTM model
python main.py --model 'pdglstm' --model_num 0
# train a PDGLSTM model with specific setting
python main.py --model 'pdglstm' --model_num 0 --batch_size 32 --init_lr 0.001 --min_lr 1e-6 --lr_decay_rate 0.1
# train a TabNet model
python main.py --graph_info False --model 'tabnet' --model_num 0
```

## Results
The test scripts of LR, RF and XGBoost are in [test/ml_test.ipynb] and their corresponding trained models are in `checkpoints/preprocessed_data/split_rs123`.  
To use the [trained neural network models], please download files and unzip it to `dl`.  
Please change the settings in [test/main_test.py] to run the test phase with different models.  
```sh
python main_test.py
```
Besides, scripts for generating figures in the main pages and supplementaries are avaliable at [models/post_aly.ipynb] and [models/post_aly_all.ipynb] respectively.


[//]: # "These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax"

[nosoi]: <https://github.com/slequime/nosoi>
[singularity]: <https://sylabs.io/guides/3.9/user-guide/>
[here]: <https://genome.ufl.edu/download/ddt/singularity_deepdynatree.sif>
[original and preprocessed datasets]: <https://genome.ufl.edu/download/ddt/data.zip>
[trained neural network models]: <https://genome.ufl.edu/download/ddt/trained_models.zip>
[SetTransformer]: <http://proceedings.mlr.press/v97/lee19d.html>
[DeepSet]: <https://doi.org/10.48550/arXiv.1703.06114>
[TabNet]: <https://doi.org/10.48550/arXiv.1908.07442>
[GCN]: <https://doi.org/10.48550/arXiv.1609.02907>
[GAT]: <https://doi.org/10.48550/arXiv.1710.10903>
[GIN]: <https://doi.org/10.48550/arXiv.1810.00826>
[dl/config.py]: <https://github.com/salemilab/DeepDynaTree/blob/main/dl/config.py>
[models/ml_models.ipynb]: <https://github.com/salemilab/DeepDynaTree/blob/main/models/ml_models.ipynb>
[test/ml_test.ipynb]: <https://github.com/salemilab/DeepDynaTree/blob/main/test/ml_test.ipynb>
[test/main_test.py]: <https://github.com/salemilab/DeepDynaTree/blob/main/test/main_test.py>
[models/post_aly.ipynb]: <https://github.com/salemilab/DeepDynaTree/blob/main/models/post_aly.ipynb>
[models/post_aly_all.ipynb]: <https://github.com/salemilab/DeepDynaTree/blob/main/models/post_aly_all.ipynb>
