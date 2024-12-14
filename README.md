# Traffic-PyTorch Execution Manual

---

### I. Introduction: Overview of Traffic-PyTorch Framework

The Traffic-PyTorch framework is a collection of cutting-edge spatiotemporal deep learning models designed for traffic prediction. It includes implementations of various prominent architectures such as ASTGCN, DCRNN, GMAN, and more. This manual provides a comprehensive guide to execute and modify the project.

---

### II. Environment Setup and Requirements

1. **System Requirements**:
    - Operating System: Linux, macOS, or Windows with WSL support.
    - Python Version: 3.8 or higher.
    - Memory: Minimum 8GB RAM (16GB recommended).
    - GPU: NVIDIA GPU with CUDA support.

2. **Dependency Installation**:
    - Clone the repository and install dependencies:
      ```bash
      git clone https://github.com/your-repo/traffic-pytorch.git
      cd traffic-pytorch
      pip install -r requirements.txt
      ```

3. **Data Preparation**:
    - Ensure the dataset follows the required format. The `data` directory includes utilities like `datasets.py` and `utils.py` to preprocess custom datasets.
    - Preloaded datasets like PEMS-Bay can be configured via `config` files.

---

### III. Project Structure

1. **Key Directories**:
    - `config`: Configuration files for various models (ASTGCN, DCRNN, etc.).
    - `data`: Utilities for dataset preprocessing and handling.
    - `evaluation`: Evaluation metrics like MAPE, RMSE, etc.
    - `logger`: Logging utilities.
    - `model`: Implementations of all supported architectures.
    - `trainer`: Training scripts for individual models.
    - `util`: Additional utilities for processing and logging.

2. **Important Files**:
    - `train.py`: Main script for training models.
    - `config/*.py`: Model-specific configurations.
    - `trainer/base_trainer.py`: Base trainer class for custom modifications.

---

### IV. Execution Steps

1. **Configuring the Model**:
    - Choose a model by modifying the corresponding configuration file in `config/`.
    - Example: Editing hyperparameters in `ASTGCN_config.py`.

2. **Training the Model**:
    - Execute `train.py` with the desired configuration:
      ```bash
      python train.py --model ASTGCN --config config/ASTGCN_config.py
      ```

3. **Evaluating the Model**:
    - Use `evaluation/metrics.py` to compute evaluation metrics:
      ```bash
      python evaluation/metrics.py --predictions results/pred.csv --ground_truth data/ground_truth.csv
      ```

4. **Visualizing Results**:
    - Leverage utilities in `logger/` to log and visualize training progress.

---

### V. Notes

1. **Custom Dataset Integration**:
    - Use `data/datasets.py` to adapt new datasets.
    - Ensure compatibility with the expected format (e.g., `.csv` or `.pkl`).

2. **Extending Models**:
    - New models can be added by defining them in the `model` directory.
    - Follow the structure of existing models like `STGCN.py`.

3. **Error Handling**:
    - Logs can be checked in the output directory specified in the configuration files.

---

### VI. Conclusion

The Traffic-PyTorch framework is a versatile tool for exploring state-of-the-art traffic prediction models. By following this manual, users can train, evaluate, and customize models to suit their specific requirements.


---

# traffic-pytorch
Integrated platform for urban intelligence tasks including traffic and demand prediction.

## Traffic prediction
We report MAE / RMSE in pems-bay dataset (12 steps / 1 hour).

![](traffic-pytorch/repo/pems-bay.png)

| Model | MAE | RMSE |
|-------|--|--|
| [DCRNN](https://openreview.net/forum?id=SJiHXGWAZ) | 0.92 | 1.58 |
| [GMAN](https://aaai.org/ojs/index.php/AAAI/article/view/5477)| 1.99 | 3.87 |
| [WaveNet](https://www.ijcai.org/proceedings/2019/264)| 4.70 | 7.53 |

We report MAE / RMSE in PeMS dataset (9 steps).

![](traffic-pytorch/repo/pems.png)

| Model | MAE | RMSE |
|-------|--|--|
| [STGCN](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17135)| 18.30 | 18.92 |
| [ASTGCN](https://ojs.aaai.org/index.php/AAAI/article/view/3881)| 2.94 | 5.50 |
| [MSTGCN](https://ojs.aaai.org/index.php/AAAI/article/view/3881)| 2.94 | 5.52 |

## Getting Started
### Data
- pems-bay, metr-la: Download h5 files from [Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) and place in datasets directory.
- PeMSD7: Download files from [STGCN Github](https://github.com/VeritasYin/STGCN_IJCAI-18) and place in datasets directory.
- PEMS: Download files from [ASTGNN Github](https://github.com/guoshnBJTU/ASTGNN) and place in datasets directory.


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
```
# DCRNN 
python train.py --model DCRNN --ddir ../datasets/ --dname pems-bay --device $DEVICE$ --num_pred 12

# GMAN
python train.py --model GMAN --ddir ../datasets/ --dname pems-bay --device $DEVICE$ --num_pred 12

# WaveNet 
python train.py --model WaveNet --ddir ../datasets/ --dname pems-bay --device $DEVICE$ --num_pred 12

# STGCN
python train.py --model STGCN --ddir ../datasets/ --dname PEMSD --device $DEVICE$ --num_pred 9

# ASTGCN
python train.py --model ASTGCN --ddir ../datasets/ --dname PEMSD --device $DEVICE$ --num_pred 9

# MSTGCN 
python train.py --model MSTGCN --ddir ../datasets/ --dname PEMSD --device $DEVICE$ --num_pred 9
```
