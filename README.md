# Modular Building Damage Detection Framework

Questa è una riorganizzazione modulare dei due notebook:

1. `create_dataset_patches_BDD.ipynb`
2. `train_fold_LF.ipynb`

## Struttura

```text
configs/
  patches.yaml              # percorsi e parametri per creare le patch
  train_late_fusion.yaml    # parametri training/evaluation
scripts/
  create_patches.py
  train_late_fusion.py
  evaluate_late_fusion.py
src/bdd/
  patches.py                # creazione patch SAR/RGB/footprint
  dataset.py                # Dataset PyTorch e stratified folds
  model.py                  # late-fusion network
  train.py                  # training cross-validation
  evaluate.py               # valutazione CV
  metrics.py                # metriche AUROC/PR/F1/Kappa
  utils.py                  # seed e init pesi
  config.py                 # lettura YAML
```

## Installazione

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

## Uso

### 1. Creare le patch

Modificare `configs/patches.yaml`, poi:

```bash
python scripts/create_patches.py --config configs/patches.yaml
```

Output atteso:

```text
dataset_patches_BDD/
  patch_list.csv
  D_<osm_id>_SAR.tif
  D_<osm_id>_RGB.tif
  D_<osm_id>_SARftp.tif
  I_<osm_id>_SAR.tif
  I_<osm_id>_RGB.tif
  I_<osm_id>_SARftp.tif
```

### 2. Addestrare late fusion

Modificare `configs/train_late_fusion.yaml`, poi:

```bash
python scripts/train_late_fusion.py --config configs/train_late_fusion.yaml
```

### 3. Valutare i fold

```bash
python scripts/evaluate_late_fusion.py --config configs/train_late_fusion.yaml
```

## Nota

Questa versione conserva la logica principale dei notebook, ma separa:

- parametri da codice;
- creazione dataset da training;
- dataset/loading da modello;
- training da evaluation;
- metriche da plotting/notebook.
