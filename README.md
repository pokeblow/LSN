# Layer Separation Network (LSN)

> **Layer Separation: Towards Adjustable Joint Space Width Images Synthesis**
> *Haolin Wang, Yafei Ou, Prasoon Ambalathankandy, Gen Ota, Pengyu Dai, Masayuki Ikebe, Kenji Suzuki, Tamotsu Kamishima.*
> Proceedings of the 33rd ACM International Conference on Multimedia (**ACM MM '25**), Dublin, Ireland.
> [doi:10.1145/3746027.3755407](https://doi.org/10.1145/3746027.3755407)

LSN is a layer-separation framework for finger-joint conventional radiography (CR). Given a single joint X-ray image with bone masks, it decomposes the image into three physically meaningful layers — **upper bone**, **lower bone** and **soft tissue** — under the radiographic imaging principle (exponential X-ray absorption). By randomly shifting the two bone layers and reconstructing the image, LSN can synthesize joint images with **adjustable joint space width (JSW)**, generating a controllable, balanced and privacy-friendly dataset for downstream rheumatoid arthritis (RA) tasks such as JSN progression quantification, JSW regression and SvdH-like JSN scoring.

## Highlights

- **Layer separation under radiographic imaging principles.** A reconstruction function `R = 1 - Π_i (1 - L_i)` ties the generated layers back to the input image, so the decomposition is physically consistent rather than purely data-driven.
- **Adversarial supervision without paired GT.** A segmentation-based supervision network `N_S` and a soft-tissue discrimination network `N_D` co-train with the generator `N_G` to remove bone shadows from the soft-tissue layer and keep bone layers correctly placed after random shifts.
- **Adjustable JSW synthesis.** Once `N_G` is trained, applying the random-shifting function `f_s` to the upper / lower bone layers and reconstructing with the soft-tissue layer yields synthetic images at arbitrary, controllable JSW from a single real image.
- **Strong downstream gains.** Pre-training ResNet-50 / ViT / Swin backbones with LSN-synthesized images consistently improves JSN progression, JSW regression and SvdH-like JSN scoring metrics, and reduces the amount of real annotated data needed.
- **Reported results (paper, MCP joints).** Reconstruction quality MSE 2.19e-4, SSIM 95.02e-2, PSNR 36.66, FID 3.03e-2; visual Turing test accuracy 0.71 averaged across 5 radiological technologists; inference 90.91 fps at 256×256 on a single GTX 2080 Ti.

## Method overview

LSN consists of five components (Fig. 2 in the paper):

| Symbol | Component | Implementation |
|---|---|---|
| `N_G` | Layer image generation network | `model/generator.py` (default backbone: TransUNet) |
| `N_S` | Segmentation-based supervision network | `model/segmenter.py` (default: U-Net) |
| `N_D` | Soft-tissue discrimination network | `model/discriminator.py` (default: U-Net) |
| `f_r` | Reconstruction function | `reconstruction_all` in `model/utils.py` |
| `f_s` | Random shifting function | `random_movement` / `random_move_and_reconstruct` in `model/utils.py` |

The training loss combines three terms (`model/lossfunction.py`):

- `L_0` — RMSE between the reconstructed image `f_r(L)` and the original joint image.
- `L_1` — BCE between the segmentation mask predicted by `N_S` on the shifted reconstruction `f_r(f_s(L))` and the shifted GT mask `f_s(M)`.
- `L_2` — dual BCE between the bone-shadow region predicted by `N_D` on the soft-tissue layer `L_0` and the GT.
- `L_3` (*pre-training only*) — RMSE between the predicted and pseudo bone layers, with and without random shifting (`LayerSegLoss_Pre`).

Training follows Algorithm 1 of the paper: a **Stage 1** pre-training on the non-overlap pseudo dataset `D1` with the auxiliary `L_3`, then **Stage 2** fine-tuning on the full real + pseudo dataset `D2` with the original LSN loss.

## Repository layout

```
LSN/
├── run.ipynb                  # Two-stage training entry-point notebook
├── summary_view.ipynb         # Helper notebook for visualizing logs
├── train_pre.py               # Stage 1: pre-training on D1 (non-overlap pseudo data)
├── train_main.py              # Stage 2: fine-tuning on D2 (real + pseudo)
├── trainer.py                 # Generic Trainer / BaseTrainModule
├── train.py                   # Standalone training script (legacy)
├── dataset.py                 # Data_Loader / Data_Loader_All / _Overlap / _Nonoverlap
├── model/
│   ├── generator.py           # N_G — wraps a segmentation backbone
│   ├── segmenter.py           # N_S — segmentation-based supervision
│   ├── discriminator.py       # N_D — soft-tissue bone-shadow discriminator
│   ├── reconstructor.py       # Reconstruction helpers
│   ├── registrater.py         # Registration helpers
│   ├── lossfunction.py        # MaskedBCE / MaskedMSE / LayerSegLoss(_Pre/_Main)
│   ├── parameter.py           # Hyper-parameters
│   ├── utils.py               # f_r, f_s, visualization
│   └── backbones/             # unet, nestedunet, denseunet, resnet50,
│                              # segnet, swinunet, transunet, esrt
├── Data/
│   ├── LS_K1_nonoverlap_train / _test       # D1, pseudo non-overlap pairs
│   ├── LS_K1_overlap_train    / _test       # synthetic overlap pairs
│   ├── LS_K1_all_train        / _test       # D2, all images (real + pseudo)
│   ├── DR_K1_train            / _test       # downstream-task split
│   └── Dataset_cases/                       # per-case image folders
├── experiments/
│   ├── utils_eval.py
│   └── Exp_downstream/
│       ├── JSN_evaluation/    # JSN progression
│       ├── JSW_evaluation/    # JSW regression
│       └── SvdH_evaluation/   # SvdH-like JSN scoring
└── logs/                      # TensorBoard / checkpoint output
```

## Installation

The code targets Python 3.10 and PyTorch ≥ 1.12 with CUDA. A typical environment can be set up with:

```bash
conda create -n lsn python=3.10 -y
conda activate lsn

# Install a CUDA build of PyTorch matching your driver, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Required Python packages
pip install numpy pandas matplotlib opencv-python pillow scipy \
            pycocotools timm tensorboard tqdm scikit-learn jupyter
```

> Hardware used in the paper: workstation with 3× NVIDIA GeForce GTX 2080 Ti. Training defaults assume CUDA is available; the code falls back to CPU if not.

## Data preparation

The dataset is derived from the MCP joint corpus of [Wang et al.](https://github.com/pokeblow/Joint-Image-Dataset): **430 MCP joints / 1,594 joint images with bone masks**, split 3:1 train:test by joint. Place (or symlink) the data under `Data/` so that the following JSON-indexed splits exist:

```
Data/LS_K1_nonoverlap_train      # D1 (used by train_pre.py)
Data/LS_K1_nonoverlap_test
Data/LS_K1_all_train             # D2 (used by train_main.py)
Data/LS_K1_all_test
Data/LS_K1_overlap_train         # synthetic overlap pairs
Data/LS_K1_overlap_test
Data/DR_K1_train / DR_K1_test    # for downstream tasks
Data/Dataset_cases/              # per-case image folders
```

The custom datasets `Data_Loader_Nonoverlap` (Stage 1) and `Data_Loader_All` (Stage 2) in `dataset.py` consume these JSON manifests directly — no extra preprocessing script is required to start training.

## Training

### Quick start (notebook)

`run.ipynb` is the recommended entry point. It executes the two stages defined in the paper:

```python
# Stage 1 — Pre-training on D1 (non-overlap pseudo data)
import runpy
runpy.run_module("train_pre", run_name="__main__")

# Stage 2 — Main training on D2 (real + pseudo)
runpy.run_module("train_main", run_name="__main__")
```

### Stage 1 — Pre-training (`train_pre.py`)

```bash
python train_pre.py
```

Defaults (matching `if __name__ == "__main__"` in `train_pre.py`):

- Image size: 256 × 256
- Optimizer: AdamW, weight decay 1e-2, β = (0.5, 0.999), `amsgrad=True`
- Initial LR: `lr_G = 1e-4`, `lr_S = 1e-4`, `lr_D = 3e-4`; `StepLR(step=100, γ=0.5)`
- Batch size: 9; epochs: 150 (paper: 300, with `m = 200`)
- Loss: `LayerSegLoss_Pre` (L0 + L1 + L2 + L3)
- Dataset: `Data/LS_K1_nonoverlap_{train,test}`
- Checkpoints written to `parameter/1_4/{generator,segmenter,discriminator}_Pre_*.pth`

### Stage 2 — Main training (`train_main.py`)

```bash
python train_main.py
```

Stage 2 expects the Stage-1 weights in `parameter/1_4/`. Update `pre_name` and `checkpoint` in `train_main.py` to point at the checkpoint you want to load (default: `pre_name = "256_(1_4)_2_5"`, `checkpoint = "_checkpoint_180"`).

Defaults:

- Image size: 256 × 256, batch size 9, epochs 100
- LR: `lr_G = 1e-4`, `lr_S = 1e-5`, `lr_D = 4e-4`; `StepLR(step=100, γ=0.5)`
- Loss: `LayerSegLoss_Main` (L0 + L1 + L2)
- Dataset: `Data/LS_K1_all_{train,test}`
- Checkpoints written to `parameter/1_4/{generator,segmenter,discriminator}_Main_*.pth`

### Backbones

`Generator`, `Segmenter` and `Discriminator` accept a `backbone` argument. Available implementations under `model/backbones/`: `unet`, `nestedunet`, `denseunet`, `resnet50`, `segnet`, `swinunet`, `transunet`, `esrt`. Defaults reproduce the paper:

```python
generator_backbone   = "transunet"
segmenter_backbone   = "unet"
discriminator_backbone = "unet"
```

## Adjustable JSW synthesis

After Stage 2, given a trained `N_G` and an input joint image `J` with bone masks `M`:

```python
layers = net_G(J, M)                     # L = (L_lower, L_upper, L_soft)
shifted = random_movement(...)           # f_s(L, t*)  with chosen / random t*
J_star = reconstruction_all(shifted)     # f_r(f_s(L, t*))
```

The shifting parameters `t*` (translation `x, y` and rotation `θ`) double as ground truth for the synthesized JSW / SvdH labels (Section 2.3 of the paper).

## Downstream evaluation

Three downstream task pipelines are provided under `experiments/Exp_downstream/`. Each follows the same `train_pre.ipynb → train_main.ipynb → evaluation.ipynb` pattern, where `train_pre` warm-starts the backbone on LSN-synthesized images and `train_main` fine-tunes on the real annotated data.

| Task | Folder | Metrics reported in the paper |
|---|---|---|
| JSN progression quantification | `JSN_evaluation/` | MSE, σ, σ′ |
| JSW regression                  | `JSW_evaluation/` | MSE, MAE, EVS, R² |
| SvdH-like JSN scoring           | `SvdH_evaluation/` | ACC, SEN, SPC, PRE |

Pre-training with LSN synthetic data improves all three tasks across ResNet-50, ViT and Swin backbones, and most of the gain is preserved when the amount of real annotated data is reduced (Fig. 4 of the paper).

## Citation

If this code or dataset helps your research, please cite the paper:

```bibtex
@inproceedings{wang2025layer,
  title     = {Layer Separation: Towards Adjustable Joint Space Width Images Synthesis},
  author    = {Wang, Haolin and Ou, Yafei and Ambalathankandy, Prasoon and Ota, Gen and Dai, Pengyu and Ikebe, Masayuki and Suzuki, Kenji and Kamishima, Tamotsu},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)},
  year      = {2025},
  address   = {Dublin, Ireland},
  publisher = {ACM},
  doi       = {10.1145/3746027.3755407}
}
```

## Contact

Questions and bug reports are welcome via GitHub Issues. For research-related questions, please contact the corresponding author **Yafei Ou** (`ou.y.ac@m.titech.ac.jp`) or the first author **Haolin Wang** (`haolin.wang.k3@elms.hokudai.ac.jp`).
