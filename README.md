# U-Net family for 4-class land cover on Sentinel-2 RGB tiles

Classes in your label files: **1 = built-up, 2 = water, 3 = trees, 4 = barren**
(remapped to 0..3 internally; padded corners of edge tiles = 255 = ignored by loss and metrics).

All models share the same data, loss, split, metrics and training loop, so they compare fairly:

| Model | `--model` | Backbone | Weights | Params |
|---|---|---|---|---|
| Normal (classic) U-Net | `unet` | 4 conv levels, widths 32-512 | from scratch | 7.76 M |
| **Attention U-Net** | `attn_unet` | same as `unet` + 4 attention gates | from scratch | 7.85 M |
| U-Net with ResNet encoder | `resnet_unet` | ResNet34 (or 18 / 50) | ImageNet-pretrained (or scratch) | 24.47 M |
| ResNet U-Net + attention gates | `attn_resnet_unet` | same as `resnet_unet` + 5 gates | ImageNet-pretrained (or scratch) | 24.60 M |

**Attention U-Net in one paragraph.** Same U-Net, but each skip connection goes through a gate first. The gate looks at the
skip features and at the (up-sampled) decoder features and outputs one map with values 0..1; the skip features are multiplied by
that map, so irrelevant background is suppressed before it reaches the decoder. `attn_unet` is *exactly* the classic U-Net
plus the gates (checked: with all gates forced open the two give identical outputs), so any difference between them is due to the gates.
The gate is computed at the skip's resolution from up-sampled decoder features (the usual practical implementation of Oktay et al. 2018).

## Files
`dataset.py` (tiles, padding, augmentation) · `model.py` (all U-Nets) · `utils.py` (loss, metrics) ·
`train.py` · `predict.py` · `check_data.py` · `summarize.py` (compare runs) · `attention_maps.py` (draw the gates)

## 1. Setup
```bash
pip install -r requirements.txt        # torch, torchvision, rasterio, numpy
```
Layout: `DATASET/{Train,Validation,Test}/{images,labels}/<same_name>.tif`.
**The ImageNet weights are downloaded automatically on the first pretrained run, so that run needs internet.**
Offline? Use `--no_pretrained`.

## 2. Check, then smoke-test (2 minutes)
```bash
python check_data.py --data_root DATASET
python train.py --data_root DATASET --model resnet_unet --limit 16 --epochs 2 --batch_size 4 --workers 0 --name smoke
```
(Smoke-test numbers mean nothing; they only prove the code runs.)

## 3. The runs
```bash
python train.py --data_root DATASET --model unet                          # 1. normal U-Net
python train.py --data_root DATASET --model attn_unet                     # 2. same + attention gates   (do the gates help?)
python train.py --data_root DATASET --model resnet_unet                   # 3. ResNet34, ImageNet-pretrained
python train.py --data_root DATASET --model attn_resnet_unet              # 4. ResNet34 + attention gates
python train.py --data_root DATASET --model resnet_unet --no_pretrained   # optional: does pretraining help?
```
Pairs to compare: 1 vs 2 (gates on the scratch U-Net), 3 vs 4 (gates on ResNet), 1 vs 3 (pretrained ResNet encoder).
Three seeds each for the final comparison:
```bash
for s in 1 2 3; do python train.py --data_root DATASET --model attn_unet --seed $s; done
python summarize.py --runs runs            # mean ± std table (test split); add --split val for validation
```
Use a GPU (Colab / Kaggle). On one CPU core an epoch takes several minutes.

## 4. Defaults (ResNet U-Net)
| Setting | Value |
|---|---|
| Input / normalisation | 3x256x256; **ImageNet mean/std** when pretrained, train-set mean/std otherwise |
| Decoder | up-sample + skip + 2 convs at 5 levels; **plus a full-resolution skip** from the raw tile, because the ResNet stem down-samples 4x at once and 10 m objects are tiny |
| Loss | class-weighted cross-entropy + Dice (weights printed at start) |
| Optimiser | AdamW; **encoder lr 1e-4, decoder lr 1e-3** (pretrained) or 1e-3 for everything (scratch); wd 1e-4; 3-epoch warm-up, cosine decay, grad-clip 1.0 |
| Stopping | up to 100 epochs, early stop on **val mIoU** (patience 20); best epoch kept; test evaluated once at the end |

Switches: `--encoder resnet18|resnet34|resnet50`, `--lr_enc`, `--lr`, `--dropout`, `--class_weights none`,
`--batch_size 8` (low GPU memory), `--workers 0` (if multiprocessing misbehaves, e.g. Windows), `--device cpu|cuda|mps`.

Outputs per run in `runs/<name>/`: `best.pt`, `history.csv` (per-epoch loss and per-class IoU), `metrics.json`.

## 5. See what the attention gates do
```bash
python attention_maps.py --checkpoint runs/<attn run>/best.pt --data_root DATASET --split Test --n 6
```
Saves `attention_viz/attention_test.png` (rows = tiles; columns = RGB | true label | prediction | attention fine to coarse;
bright = skip features let through) and prints the mean attention on each true class. Use it only on a *trained* model;
an untrained one shows ~0.5 everywhere. Don't over-read the pictures: attention maps are a hint, not proof of what the model "thinks".

## 6. Apply to new areas
```bash
python predict.py --checkpoint runs/<name>/best.pt --input new_tiles/ --out preds/ --crs EPSG:32643 --tta
```
Writes one GeoTIFF per input (values 1-4) and `area_ha.csv` (hectares per class; needs a metre grid).
Your files say EPSG:4326 but the coordinates look like UTM (probably zone 43N): confirm, then pass `--crs`.
Works for big mosaics (sliding window, 64 px overlap). Works with checkpoints of either model.

## 7. Reading the results
* Compare **per-class IoU**, especially barren (precision and recall); it drives the solar siting.
* Expect built-up vs barren confusion (RGB only).
* The Test set is weak (50 tiles; water < 1% of its pixels), so test water IoU is noisy. Select on Validation; don't tune on Test.
* Differences of 1-2 points between models are within seed noise; use 3 seeds.

## 8. Next
NIR/SWIR bands + indices (NDVI, NDBI, MNDWI/NDWI, BSI), re-exported on the same tiles and date as the RGB, then repeat the comparison.
