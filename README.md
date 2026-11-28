# Thermal Image Super-Resolution · 8×

**Experimental implementations for reconstructing detail in low-resolution thermal images.**

This project explores 8× thermal super-resolution through CNN baselines, residual networks, and GAN-based models, with experiments in edge-aware reconstruction and frequency-domain supervision.

## Explore the repository

| Resource | What you'll find |
| --- | --- |
| [`models/`](models/) | Training scripts for SRCNN, VDSR-style, SRResNet, SRGAN, and Real-ESRGAN-style variants |
| [`inference/`](inference/) | Model-specific inference and evaluation scripts |
| [`output_samples/`](output_samples/) | Saved output images grouped by experiment |
| [`thermal_DINO_loss.py`](thermal_DINO_loss.py) | An additional feature-loss experiment |
| [Detailed technical guide](README%281%29.md) | Architecture notes, loss formulations, configuration, and experimental workflow |
| [Project report](Physics_Aware_Thermal_Super_Resolution.pdf) | The included thermal super-resolution report |

## Technical focus

- Compare L1 and L2 reconstruction objectives in shallow CNN baselines.
- Explore residual upsampling and hybrid reconstruction losses.
- Incorporate Sobel-derived edge information.
- Investigate FFT-based frequency supervision.
- Examine GAN training strategies adapted to thermal imagery.

These are experimental directions represented by the code. The saved examples are qualitative artifacts; they do not establish general benchmark superiority.

## A starting point

1. Read the [technical guide](README%281%29.md) and choose a model family.
2. Inspect the matching script in [`models/`](models/) for dataset locations, dependencies, and training settings.
3. Configure paths for your own low/high-resolution thermal image pairs.
4. Use the corresponding [inference script](inference/) with a compatible checkpoint.
5. Compare outputs using a fixed validation set, PSNR/SSIM, and visual inspection of thermal boundaries.

The repository does not include a complete dataset or trained checkpoint bundle. Some scripts use local paths, so configuring your environment is part of reproducing an experiment.

## Lightweight CPU checks

With PyTorch installed, run from the repository root:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

The thirteen checks cover importing the SRCNN L1 inference model without checkpoint
reads, output-directory creation, device probing, or optional image packages; a
tiny CPU forward pass with finite output and preserved image dimensions; CLI
argument handling and synthetic checkpoint loading onto CPU; file-pair discovery
with early rejection of missing or empty inputs; PSNR values for known errors and
exact matches; and matching tensor shapes for PSNR/SSIM inputs.
Importing `SRCNN` from `inference.srcnn_l1_inference` now only defines the model
and helpers. Running the file directly requires explicit input/checkpoint paths as shown below,
plus Torchvision, Pillow, and `pytorch_msssim` (for SSIM). Other inference scripts have not been refactored.
These checks do not validate trained reconstruction quality or full evaluation.

### Automated CPU checks

GitHub Actions runs the thirteen regression checks on pushes to `main`, pull requests,
and manual dispatch. The workflow uses Ubuntu 24.04, Python 3.13, CPU-only PyTorch
2.12.1, read-only repository permissions, pinned action revisions, and a ten-minute
timeout. It does not require thermal images, trained weights, or a GPU.

To reproduce its direct dependency in a fresh Python 3.13 virtual environment:

```bash
python -m pip install torch==2.12.1+cpu --index-url https://download.pytorch.org/whl/cpu
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Transitive dependencies are resolved by pip rather than locked. This minimal
environment covers SRCNN import, argument handling, synthetic weights, forward
execution, and file discovery; it does not include the optional image-evaluation
packages or establish that full inference or other model families work.

## SRCNN L1 inference command

```bash
python inference/srcnn_l1_inference.py \
  --lq-dir /path/to/LQ \
  --gt-dir /path/to/HR \
  --checkpoint /path/to/best_SRCNN.pth \
  --output-dir /path/to/new-results \
  --device cpu
```

The three input paths are required. `--output-dir` defaults to `srcnn_l1_output`;
use a separate output directory to preserve inputs and previous results.
`--device` accepts `cpu`, `cuda`, or `auto` (the default, choosing CUDA if available).
Checkpoint tensors are mapped to the selected device and loaded as a state
dictionary with `weights_only=True`. `--help` works without optional image packages.

The original preprocessing still resizes low-resolution inputs to 448 × 640;
ground-truth images must have that size and matching filenames. Supply nonempty
folders containing image pairs. Discovery accepts PNG, JPG/JPEG, BMP, and TIF/TIFF
extensions case-insensitively, ignores other files and subdirectories, and sorts
filenames. Every selected low-resolution file must have an exact filename match
in the ground-truth folder; extra ground-truth files are unused. Missing folders,
empty selections, and missing pairs fail before optional imports, checkpoint
loading, or output-directory creation. File contents and dimensions are still
checked only when processed; the discovery tests use placeholder files.
This change does not alter the model architecture,
normalization, metrics, or saved image format. Full dataset evaluation has not
been verified by the synthetic tests.

### PSNR convention

SRCNN L1 evaluates PSNR with a fixed data range of 1 after denormalizing and
clamping its output to `[0, 1]`: `PSNR = -10 * log10(MSE)`. Exact matches return
positive infinity, rather than the previous artificial 100 dB cap. Consequently,
the arithmetic mean printed by the script is also infinite if any evaluated
pair is an exact match. Nonzero-error scores retain the same mathematical
formula. PSNR and SSIM require prediction and target tensors to have identical
shapes; mismatches raise `ValueError` instead of broadcasting. This correction
applies only to `srcnn_l1_inference.py`; saved results have not been recomputed,
and other model-specific metric helpers are unchanged.

## Sample output

An output from the edge-injection experiment:

![Edge-injection thermal super-resolution output](output_samples/edge_injection/edge_output_001_01_D3_th.png)

Browse the [output gallery](output_samples/) to compare saved artifacts across model families. Use matched inputs and documented configurations when making quantitative comparisons.

## Next milestones

- [ ] Consolidate dataset paths and model settings into shared configuration.
- [ ] Add a tested dependency specification.
- [ ] Publish comparable input/output panels and evaluation settings.
- [ ] Document checkpoint availability and reproducibility for each experiment.

A collaborative project by [Priyanshi Goyal](https://github.com/drain-is-she) and [Vansh Joshi](https://github.com/Vansh-A1).

