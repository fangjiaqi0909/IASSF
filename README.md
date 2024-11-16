# Infrared-Assisted Single-Stage Framework for Joint Restoration and Fusion of  Visible and Infrared Images under Hazy Conditions

1. Install PyTorch 1.10.2 and torchvision 0.11.3.
    ```bash
    conda install -c pytorch pytorch torchvision
    ```

2. Install the other dependencies.
    ```bash
    pip install -r requirements.txt
    ```
    
## Prepare

The final file path should be the same as the following (please check it carefully):
```
┬─ save_models
│   ├─ iassf
│   │   ├─ IASSF.pth
│   │   └─ ... (model name)
│   └─ ... (exp name)
└─ data
    ├─ MSRS
    │   ├─ train
    │   │   ├─ GT
    │   │   │   └─ ... (image filename)
    │   │   └─ hazy
    │   │   │   └─ ... (image filename)
    │   │   └─ IR
    │   │       └─ ... (image filename)
    │   └─ test
    │   │   ├─ GT
    │   │   │   └─ ... (image filename)
    │   │   └─ hazy
    │   │   │   └─ ... (image filename)
    │   │   └─ IR
    │   │       └─ ... (image filename)   
    └─ ... (dataset name)
```

## Training

To customize the training settings for each experiment, navigate to the `configs` folder. Modify the configurations as needed.

After adjusting the settings, use the following script to initiate the training of the model:

```sh
CUDA_VISIBLE_DEVICES=X python train.py --model (model name) --dataset (dataset name) --exp (exp name)
```

For example, we train the IASSF on the MSRS:

```sh
CUDA_VISIBLE_DEVICES=0 python test.py --model iassf --dataset MSRS --exp iassf
```

## Testing

Run the following script to evaluate the trained model with a single GPU.


```sh
CUDA_VISIBLE_DEVICES=X python test.py --model (model name) --dataset (dataset name) --exp (exp name)
```

For example, we test the IASSF on the MSRS:

```sh
CUDA_VISIBLE_DEVICES=0 python test.py --model iassf --dataset MSRS --exp iassf
```

