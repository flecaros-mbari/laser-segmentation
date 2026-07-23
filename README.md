## Laser segmentation

Program to train and test a U-Net for my master degree.

### Codes in this repository

* `U_Net_highresolution_nonilluminated.ipynb` - the original Colab notebook.
* `laser_segmentation/` - the same pipeline (model, dataset, losses, metrics)
  as a local Python package, plus `train.py` / `predict.py` / `evaluate.py`
  as command-line entry points, so it can run outside Colab.

### Setup

```
pip install -r requirements.txt
```

### Usage

Train a model (images and masks must share filenames between the two directories):

```
python train.py \
    --image-dir data/images --mask-dir data/masks \
    --output-dir runs/highres --test-name highres \
    --image-width 1920 --image-height 1080 --epochs 50 --batch-size 5
```

This writes `runs/highres/highres.pt` (model weights), `highres_meta.json`
(the hyperparameters `predict.py` needs to rebuild the model), a loss plot,
and the train/test/validation file-path splits.

Run inference with a trained model:

```
python predict.py \
    --model-path runs/highres/highres.pt \
    --image-dir data/new_images --mask-dir data/new_masks \
    --output-dir runs/highres/predictions
```

`--mask-dir` is optional - omit it to run inference on unlabeled images.

Compare predictions against ground truth and record metrics in Excel
(run once per comparison you want in the report; each call adds a sheet):

```
python evaluate.py \
    --mask-dir runs/highres/predictions/mask \
    --predicted-dir runs/highres/predictions/predicted_0.15 \
    --output-xlsx runs/highres/results.xlsx --sheet-name unet_0.15
```

### TODO

* Search for other metrics to compare results
* Train with more data (different light)

### Flow of the code (general)

1. Start the UNet class
2. Define the loss function and the weight of the classes 
3. Train with the images and their ground truth
4. Save the results
5. Evaluate the results and save them in excel 
