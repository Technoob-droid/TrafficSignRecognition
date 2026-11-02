# Traffic Sign Recognition using CNN and Keras

Deep learning pipeline trained on the GTSRB dataset with 43 traffic sign classes.

## Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training`
``powershell
python src\train.py
```

## Inference
``powershell
python src\predict.py <path_to_image>
```
