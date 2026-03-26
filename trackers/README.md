# Trackers

This directory contains trackers integrated into the 360VOT omnidirectional
tracking benchmark.

---

## SiamFC (omnidirectional variant)

**SiamFC** ([Bertinetto et al., 2016](https://arxiv.org/abs/1606.09549)) is a
Siamese network tracker based on a lightweight AlexNet backbone and a simple
cross-correlation head.  It is the fastest model variant in the siamX family
due to the absence of an RPN head or feature pyramid.

### Directory structure

```
trackers/
  siamfc_omni.py           ← SiamFCOmniTracker (clean 360° wrapper class)
  siamX/
    siamx_models/          ← AlexNet backbone + XCorr head + SiamFC builder
    siamx_fc/              ← Crops, SiameseNet, SiamFCTracker, configs
    siamx_rpn/             ← Stubs (not used for SiamFC inference)
    pretrained_weights/    ← Place SiamFC.pth here (see below)
```

### Pretrained weights

SiamFC was originally trained on the ImageNet Video Detection dataset.
Download a compatible pretrained `SiamFC.pth` checkpoint and place it at:

```
trackers/siamX/pretrained_weights/SiamFC.pth
```

Several public pretrained checkpoints are available, for example from the
[PySOT model zoo](https://github.com/STVIR/pysot/blob/master/MODEL_ZOO.md)
or from the
[SiamFC PyTorch implementations](https://github.com/huanglianghua/siamfc-pytorch).

> **Weight compatibility**: The bundled AlexNet backbone expects a state-dict
> whose keys follow the pattern `backbone.features.{index}.*`.  If your
> checkpoint uses different key names, edit `trackers/siamX/siamx_models/utils.py`
> to add the required key remapping.

### Running the 360° tracker

```bash
# Track a single sequence
python scripts/run_siamfc_360.py \
    --dataset_dir /path/to/360VOT \
    --sequence    0001 \
    --output_dir  results/SiamFC \
    --model_path  trackers/siamX/pretrained_weights/SiamFC.pth

# Track all sequences
python scripts/run_siamfc_360.py \
    --dataset_dir /path/to/360VOT \
    --output_dir  results/SiamFC \
    --model_path  trackers/siamX/pretrained_weights/SiamFC.pth

# Use GPU
python scripts/run_siamfc_360.py \
    --dataset_dir /path/to/360VOT \
    --output_dir  results/SiamFC \
    --model_path  trackers/siamX/pretrained_weights/SiamFC.pth \
    --device cuda

# Live visualisation (single sequence)
python scripts/run_siamfc_360.py \
    --dataset_dir /path/to/360VOT \
    --sequence    0001 \
    --output_dir  results/SiamFC \
    --model_path  trackers/siamX/pretrained_weights/SiamFC.pth \
    --vis
```

### Interactive video demo (MP4 input)

Use `scripts/test_video_siamx.py` to test the SiamX-360 tracker on any MP4
video.  The script opens the first frame for manual bounding-box annotation,
then runs real-time tracking with live visualisation, saves an output video,
and reports inference speed.

```bash
# Basic usage — draw bbox on frame 1, then track
python scripts/test_video_siamx.py \
    --video /path/to/video.mp4 \
    --model_path trackers/siamX/pretrained_weights/SiamFC.pth

# Use GPU and specify output path
python scripts/test_video_siamx.py \
    --video /path/to/video.mp4 \
    --model_path trackers/siamX/pretrained_weights/SiamFC.pth \
    --device cuda \
    --output results/demo_out.mp4

# Headless mode (no GUI window, just save output)
python scripts/test_video_siamx.py \
    --video /path/to/video.mp4 \
    --model_path trackers/siamX/pretrained_weights/SiamFC.pth \
    --no_show
```

### Evaluating results

After running the tracker, evaluate with the standard 360VOT benchmark script:

```bash
python scripts/eval_360VOT.py \
    --dataset_dir /path/to/360VOT \
    --bfov_dir    results/SiamFC
```

### Dataset directory structure

```
<dataset_dir>/
  <sequence>/
    img/          ← frame images (e.g. 00001.jpg, 00002.jpg, ...)
    label.json    ← per-frame annotations with "bfov" key
```

### Output format

Each result file `results/SiamFC/<sequence>.txt` contains one line per frame:

```
clon,clat,fov_h,fov_v,rotation
```

### Using the Python API

```python
import cv2
from lib.utils import dict2Bfov
from trackers.siamfc_omni import SiamFCOmniTracker

tracker = SiamFCOmniTracker(
    model_path='trackers/siamX/pretrained_weights/SiamFC.pth',
    img_w=1920, img_h=960,
    search_scale=2.0,   # search region = 2× predicted BFoV
    device='cpu',
)

frames = [cv2.imread(p) for p in sorted_frame_paths]
init_bfov = dict2Bfov({'clon': 10, 'clat': 5,
                        'fov_h': 20, 'fov_v': 15, 'rotation': 0})
tracker.initialize(frames[0], init_bfov)

for frame in frames[1:]:
    pred_bfov = tracker.track(frame)
    print(pred_bfov.todict())
```

### Key hyperparameters

| Parameter       | Default | Description                                      |
|-----------------|---------|--------------------------------------------------|
| `search_scale`  | 2.0     | BFoV scale for the search region                 |
| `crop_size`     | 500     | Horizontal resolution of perspective crops       |
| `scale_num`     | 3       | Number of scales searched each frame             |
| `scale_step`    | 1.0375  | Scale step between consecutive scale levels      |
| `scale_lr`      | 0.59    | Learning rate for target size update             |
| `window_influence` | 0.176 | Cosine window weight in the response map       |
