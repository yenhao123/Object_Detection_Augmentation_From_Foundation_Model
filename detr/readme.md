## Prerequisite
conda create -n detr 
conda activate detr
conda install -c pytorch pytorch torchvision
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install torchmetrics[detection]

## Parameter Setting
arguments
* dataset_path
* output_dir
* epoch


## Training
```shell
1. data 放在 data/cvpdl/ 中
2. sh run.sh
```
>若要切換 backbone 訓練，請執行 sh run_res101.sh

## Test
```shell
python inference.py
python evaluate.py
```
>上述會自動使用最好的 model，如要更改請改參數 pretrained_path
