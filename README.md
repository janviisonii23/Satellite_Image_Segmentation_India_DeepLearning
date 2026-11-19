# Land Cover Segmentation with U-Net Variants (Sentinel-2 RGB)

This project trains and compares four U-Net models for 4-class land cover mapping on Sentinel-2 RGB tiles. The end goal is finding barren land for solar site selection, so barren-class IoU is the number I care about most.

Classes in the label files: 1 = built-up, 2 = water, 3 = trees, 4 = barren. They are remapped to 0-3 internally. Padded corners of edge tiles are set to 255 and ignored by the loss and the metrics.

All four models use the same data, loss, split, metrics and training loop, so the comparison is fair.

## Models

| Model                    | `model`            | Backbone                           | Weights                          | Params  |
| ------------------------ | ------------------ | ---------------------------------- | -------------------------------- | ------- |
| U-Net                    | `unet`             | 4 conv levels, widths 32-512       | from scratch                     | 7.76 M  |
| Attention U-Net          | `attn_unet`        | same as `unet` + 4 attention gates | from scratch                     | 7.85 M  |
| ResNet U-Net             | `resnet_unet`      | ResNet34 (or 18 / 50)              | ImageNet-pretrained (or scratch) | 24.47 M |
| ResNet U-Net + attention | `attn_resnet_unet` | same as `resnet_unet` + 5 gates    | ImageNet-pretrained (or scratch) | 24.60 M |

![Architectures of the four models](<assets/Four U-Net Architectures Compared.png>)

### How the attention gate works

In the Attention U-Net, every skip connection passes through a gate before it reaches the decoder. The gate takes the skip features and the up-sampled decoder features and produces a single map with values between 0 and 1. The skip features are multiplied by this map, which suppresses background that is not useful for the current prediction.

`attn_unet` is the classic U-Net plus the gates and nothing else. I checked this: with all gates forced open, the two models give identical outputs. So any difference between them comes from the gates. The gate is computed at the skip's resolution from up-sampled decoder features, which is the usual practical implementation of Oktay et al. (2018).

## Project structure

```
Satellite_Image_Segmentation/
├── README.md
├── requirements.txt
├── assets/
│   └── Four U-Net Architectures Compared.png
├── notebooks/
├── scripts/
│   ├── check_data.py
│   ├── predict.py
│   ├── summarize.py
│   └── train.py
└── src/
    ├── attention_maps.py
    ├── dataset.py
    ├── model.py
    └── utils.py
```

## Files

- `src/dataset.py`: tiles, padding, augmentation
- `src/model.py`: all four models
- `src/utils.py`: loss and metrics
- `scripts/train.py`: training
- `scripts/predict.py`: inference on new areas
- `scripts/check_data.py`: sanity checks on the dataset
- `scripts/summarize.py`: compare runs
- `src/attention_maps.py`: visualise the gates

## 1. Setup

```bash
pip install -r requirements.txt   # torch, torchvision, rasterio, numpy
```

Expected folder layout:

```
DATASET/{Train,Validation,Test}/{images,labels}/<same_name>.tif
```

ImageNet weights are downloaded automatically the first time you run a pretrained model, so that run needs internet. If you are offline, add `--no_pretrained`.

## 2. Check the data and run a smoke test

```bash
python scripts/check_data.py --data_root DATASET
python scripts/train.py --data_root DATASET --model resnet_unet --limit 16 --epochs 2 --batch_size 4 --workers 0 --name smoke
```

The smoke test takes about 2 minutes. Its metrics mean nothing; it only confirms the code runs end to end.

## 3. Training runs

```bash
python scripts/train.py --data_root DATASET --model unet                          # 1. plain U-Net
python scripts/train.py --data_root DATASET --model attn_unet                     # 2. U-Net + attention gates
python scripts/train.py --data_root DATASET --model resnet_unet                   # 3. ResNet34, ImageNet-pretrained
python scripts/train.py --data_root DATASET --model attn_resnet_unet              # 4. ResNet34 + attention gates
python scripts/train.py --data_root DATASET --model resnet_unet --no_pretrained   # optional: effect of pretraining
```

Useful comparisons:

- 1 vs 2: do the gates help a from-scratch U-Net?
- 3 vs 4: do the gates help the ResNet U-Net?
- 1 vs 3: does a pretrained ResNet encoder help?

For the final numbers, run each model with three seeds and summarise:

```bash
for s in 1 2 3; do python scripts/train.py --data_root DATASET --model attn_unet --seed $s; done
python scripts/summarize.py --runs runs            # mean ± std on the test split; add --split val for validation
```

Use a GPU (Colab or Kaggle works). On a single CPU core one epoch takes several minutes.

## 4. Default settings (ResNet U-Net)

| Setting               | Value                                                                                                                                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Input / normalisation | 3x256x256. ImageNet mean/std when pretrained, train-set mean/std otherwise                                                                                                                                  |
| Decoder               | Up-sample + skip + 2 convs at 5 levels, plus a full-resolution skip from the raw tile. The ResNet stem down-samples 4x at once, and objects at 10 m resolution are tiny, so this skip keeps the fine detail |
| Loss                  | Class-weighted cross-entropy + Dice (weights are printed at the start of training)                                                                                                                          |
| Optimiser             | AdamW. Encoder lr 1e-4 and decoder lr 1e-3 when pretrained, 1e-3 for everything when training from scratch. Weight decay 1e-4, 3-epoch warm-up, cosine decay, gradient clipping at 1.0                      |
| Stopping              | Up to 100 epochs, early stopping on validation mIoU (patience 20). The best epoch is kept and the test set is evaluated once at the end                                                                     |

Command-line switches:

- `--encoder resnet18|resnet34|resnet50`
- `--lr_enc`, `--lr`, `--dropout`
- `--class_weights none`
- `--batch_size 8` if GPU memory is low
- `--workers 0` if multiprocessing causes trouble (for example on Windows)
- `--device cpu|cuda|mps`

Each run writes its outputs to `runs/<name>/`:

- `best.pt`: best checkpoint
- `history.csv`: per-epoch loss and per-class IoU
- `metrics.json`: final metrics

## 5. Visualising the attention gates

```bash
python src/attention_maps.py --checkpoint runs/<attn run>/best.pt --data_root DATASET --split Test --n 6
```

This saves `attention_viz/attention_test.png` and prints the mean attention on each true class. Each row is one tile, and the columns are RGB, true label, prediction, then the attention maps from fine to coarse. Bright areas mean the skip features are let through.

Only use this on a trained model. An untrained one shows roughly 0.5 everywhere. Also, attention maps are a hint and not proof of what the model is relying on, so don't read too much into them.

## 6. Predicting on new areas

```bash
python scripts/predict.py --checkpoint runs/<name>/best.pt --input new_tiles/ --out preds/ --crs EPSG:32643 --tta
```

This writes one GeoTIFF per input tile (values 1-4) and an `area_ha.csv` with hectares per class. Area calculation needs a grid in metres.

The input files are tagged EPSG:4326, but the coordinates look like UTM (probably zone 43N). Confirm this before use, then pass the correct code with `--crs`.

Large mosaics work too, since prediction uses a sliding window with 64 px overlap. It works with checkpoints from any of the four models.

## 7. Reading the results

- Compare per-class IoU, especially barren (both precision and recall), since it drives the solar siting.
- Expect some confusion between built-up and barren, because the input is RGB only.
- The test set is small (50 tiles) and water is under 1% of its pixels, so test water IoU is noisy. Select models on Validation and do not tune on Test.
- A difference of 1-2 points between models is within seed noise, which is why I use 3 seeds.

## 8. Future work

Add NIR and SWIR bands and spectral indices (NDVI, NDBI, MNDWI/NDWI, BSI), exported on the same tiles and dates as the RGB, then repeat the comparison.
