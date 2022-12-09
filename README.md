# traffic-pytorch
Integrated platform for urban intelligence tasks including traffic and demand prediction.

## Traffic prediction
We report MaskedMSE / MaskedMAE and provide pretrained file for each prediction steps.
| Model | pems-bay (15 min) | pems-bay (30 min) | pems-bay (1 hour) |
|-------|--|--|--|
| [DCRNN](https://openreview.net/forum?id=SJiHXGWAZ) | | | |
| [GMAN](https://aaai.org/ojs/index.php/AAAI/article/view/5477)| 9.18 / 1.46 [Pretrained]() | | |
| [WaveNet](https://www.ijcai.org/proceedings/2019/264)| | | |
|[STDEN](https://ojs.aaai.org/index.php/AAAI/article/view/20322)| | | |
| [STGCN](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17135)| | | |

## Demand


## Getting Started
### Data
- pems-bay, metr-la: Download h5 files from [Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) and place in datasets directory.
- PeMSD7: Download files from [STGCN Github](https://github.com/VeritasYin/STGCN_IJCAI-18) and place in datasets directory.
- PEMS04: Download files from [STGODE Github](https://github.com/square-coder/STGODE) and [CorrSTN Github](https://github.com/drownfish19/corrstn), rename distance.csv to PEMS04_distance.csv and place in datasets directory.

### Environment
``` 
conda create -n $ENV_NAME$ python=3.7
conda activate $ENV_NAME$

# CUDA 11.3
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 
# Or, CUDA 10.2 
pip install torch==1.11.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102 
pip install -r requirements.txt
```

### Train
If config file not specified, load $MODEL_NAME$_config.py by default. 
```
python train.py --model $MODEL_NAME$ --ddir $PATH_TO_DATASET$ --dname $DATASET_NAME$
```
