## Laser segmentation

A U-Net that segments the laser line out of underwater camera images, built
for my master's degree. Given an image, it predicts a binary mask of where
the laser line falls.

### Layout

```
laser_segmentation/   the model, dataset, loss, and metrics code (a plain Python package)
scripts/               train.py / predict.py / evaluate.py - command-line entry points
notebooks/             the original Colab notebook this pipeline was ported from
requirements.txt
```

`laser_segmentation/` has no CLI or hardcoded paths in it - it's just the
reusable pieces. Each script under `scripts/` wires those pieces together
for one step of the pipeline and takes its inputs as command-line arguments.

### Setup

```
pip install -r requirements.txt
```

### Usage

**1. Train** a model. Images and masks must share filenames between the two
directories (e.g. `data/images/frame001.png` and `data/masks/frame001.png`):

```
python scripts/train.py \
    --image-dir data/images --mask-dir data/masks \
    --output-dir runs/highres --test-name highres \
    --image-width 1920 --image-height 1080 --epochs 50 --batch-size 5
```

This writes to `runs/highres/`:
* `highres.pt` - model weights
* `highres_meta.json` - the hyperparameters `predict.py` needs to rebuild the model
* `highres.png` - training/test loss curve
* `highres_{train,test,val}_{images,masks}.txt` - the file-path splits used

**2. Predict.** Run the trained model over a directory of images:

```
python scripts/predict.py \
    --model-path runs/highres/highres.pt \
    --image-dir data/new_images --mask-dir data/new_masks \
    --output-dir runs/highres/predictions
```

`--mask-dir` is optional - omit it to run inference on unlabeled images. This
also saves a fast HSV-threshold baseline mask for comparison, alongside the
U-Net's prediction.

**3. Evaluate.** Compare predictions against ground truth and record metrics
(precision/recall/F1, Hausdorff distance, pixel norms, ...) in Excel. Run it
once per comparison you want in the report - each call adds a sheet to the
same workbook:

```
python scripts/evaluate.py \
    --mask-dir runs/highres/predictions/mask \
    --predicted-dir runs/highres/predictions/predicted_0.15 \
    --output-xlsx runs/highres/results.xlsx --sheet-name unet_0.15

python scripts/evaluate.py \
    --mask-dir runs/highres/predictions/mask \
    --predicted-dir runs/highres/predictions/baseline_threshold \
    --output-xlsx runs/highres/results.xlsx --sheet-name baseline
```

### TODO

* Search for other metrics to compare results
* Train with more data (different light)
