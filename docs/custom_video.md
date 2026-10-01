# Processing Custom Video

Pre-processed examples are available from [this file](https://huggingface.co/nvidia/CARI4D/blob/main/generated-videos.zip). Download and unzip:

```bash
unzip generated-videos.zip -d data/videogen
bash scripts/demo-custom.sh data/cari4d-demo/videogen/videos/Date03_Sub01_Suitcase_Dragging-wild.0.color.mp4
```

For your own RGB videos, see `data/cari4d-demo/` for the expected data layout.

> **Note:** Our method is not designed for partially visible bodies or long-term occlusions. It works best when both the person and object are mostly visible. See the teaser videos on our [website](https://nvlabs.github.io/CARI4D/) and [generated videos](https://huggingface.co/nvidia/CARI4D/blob/main/generated-videos.zip) for examples.

---

## Step 1: Human and Object Masks

Prepare masks as a packed HDF5 file. Each frame needs a human mask and an object mask.

### Output format

`<masks_root>/<seq>_masks_k<kid>.h5` — HDF5 file with a top-level group named after the sequence (e.g. `Date03_Sub01_gas_wild002`). Each frame contributes two datasets:

| Dataset key | Type | Shape | Description |
|---|---|---|---|
| `<frame_id>-k<kid>.person_mask.png` | `bool` | `(H, W)` | Human binary mask |
| `<frame_id>-k<kid>.obj_rend_mask.png` | `bool` | `(H, W)` | Object binary mask |

- `<frame_id>` is a 6-digit zero-padded frame index (`000000`, `000001`, ...)
- For in-the-wild videos, use `kid=0`
- Masks are loaded by [this function](https://github.com/NVlabs/CARI4D/blob/main/behave_data/behave_video.py#L23-L45)

### Using SAM3 (recommended)

We provide `prep/run_sam3_masks.py` which uses [SAM3](https://github.com/facebookresearch/sam3) for text-prompted video segmentation. It takes a video and two text prompts (human + object), segments and tracks both across all frames, and saves the result in the HDF5 format above.

**Setup** (requires Python 3.12+, PyTorch 2.7+, CUDA — use a separate env from CARI4D):

```bash
# 1. Clone SAM3 into project root
git clone https://github.com/facebookresearch/sam3.git

# 2. Install SAM3 and dependencies (in a Python 3.12+ env)
cd sam3 && pip install -e . && pip install einops h5py opencv-python pycocotools psutil imageio
cd ..

# 3. Authenticate with HuggingFace for checkpoint access
#    Request access at https://huggingface.co/facebook/sam3, then:
huggingface-cli login --token $HF_TOKEN
#    Checkpoints (sam3.pt from facebook/sam3) are auto-downloaded on first run.
```

**Example:**

```bash
python prep/run_sam3_masks.py \
    --video data/cari4d-demo/wild/videos/Date03_Sub01_gas_wild002.0.color.mp4 \
    --human_prompt "man" \
    --object_prompt "a red gas cylinder" \
    --visualize
```

**Notes:**
- Processes video in chunks (default 300 frames, configurable via `--chunk_size`) to fit within 24GB GPU memory.
- Use short text prompts (e.g. `"man"`, `"person"`) for the human — long descriptive prompts may fail detection in some chunks.
- Add `--visualize` to save a side-by-side MP4 (RGB | RGB + mask overlay) for inspection.

---

## Step 2: Object Reconstruction

Use Hunyuan3D or SAM3D to reconstruct the object mesh. Place the output under `<hy3d_root>`. Step by step instruction:

- Extract first frame RGB and object mask, remove the background, and crop a square around the object with 0.20 border margin, save the result image as one RGBA png file. Note if object is heavily occluded in the first frame, we recommend using another frame in the image for better object reconstruction. 
- Run Hunyuan3D/SAM3D with the RGBA image as input,  reconstruct 3D and convert glb to obj file. 
- **Mesh path convention:** `<seq>*_<frame_index:03d>_rgba/<seq>*_<frame_index:03d>_align.obj`
  where `frame_index` is the video frame used for reconstruction.
- The mesh should be in normalized scale (longest axis in `[-1, 1]`,  direct Hunyuan3D output result). Metric-scale estimation is done later using UniDepth.

### Example with `run_hy3d_recon.py` 

We provide `prep/run_hy3d_recon.py` which automates the full pipeline: RGBA extraction from video + masks, Hunyuan3D shape and texture generation, and GLB-to-OBJ conversion.

**Setup** (use a separate env from CARI4D, requires diffusers 0.31.0):

```bash
# 1. Create and activate a Hunyuan3D environment
conda create -n hy3d python=3.10 && conda activate hy3d

# 2. Install Hunyuan3D-2 (follow https://github.com/Tencent/Hunyuan3D-2)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/Tencent/Hunyuan3D-2.git
# Or clone and install locally:
# git clone https://github.com/Tencent/Hunyuan3D-2.git && cd Hunyuan3D-2 && pip install -e .

# 3. Install additional dependencies
pip install h5py opencv-python

# 4. Ensure Blender 3.6+ is available for GLB-to-OBJ conversion
#    Download from https://www.blender.org/download/ if needed
```

**Example:**

```bash
python prep/run_hy3d_recon.py \
    --video data/cari4d-demo/wild/videos/Date03_Sub01_gas_wild002.0.color.mp4 \
    --masks_root data/cari4d-demo/wild/masks \
    --hy3d_root data/cari4d-demo/meshes \
    --blender_path /path/to/blender
```

**Notes:**
- By default uses frame 0 for reconstruction. If the object is heavily occluded in the first frame, specify `--frame_index N` to use a different frame.
- Use `--skip_hy3d` to only extract the RGBA image (e.g. for manual inspection before running reconstruction).
- Use `--skip_glb2obj` to skip the Blender conversion step (e.g. if you want to inspect the GLB first).
- The script skips processing if the output OBJ already exists.

### Alternative: `run_sam3d_recon.py` (SAM 3D Objects)

`prep/run_sam3d_recon.py` takes the same arguments and writes the same layout, with
[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) as the reconstructor.
It is the better choice for objects seen from one side in clutter (a pot on a stove, a pan
in a hand): Hunyuan3D invents the far side, SAM 3D was trained on occluded real photographs.
It also keeps the predicted camera-frame pose beside the mesh as `<name>_sam3d_pose.npz`.

**Setup** (its own env; needs a GPU with 32 GB or more):

```bash
git clone https://github.com/facebookresearch/sam-3d-objects && cd sam-3d-objects
conda env create -f environments/default.yml && conda activate sam3d-objects
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]' && pip install -e '.[p3d]'
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]' && ./patching/hydra
pip install 'huggingface-hub[cli]<1.0' h5py opencv-python
hf download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects
mv checkpoints/hf-download/checkpoints checkpoints/hf && rm -rf checkpoints/hf-download
```

Build the env on a GPU node, or PyTorch3D compiles without CUDA. The checkpoints are gated:
accept the license on HuggingFace and `hf auth login` first.

**Example:**

```bash
python prep/run_sam3d_recon.py \
    --video data/cari4d-demo/wild/videos/Date03_Sub01_gas_wild002.0.color.mp4 \
    --masks_root data/cari4d-demo/wild/masks \
    --hy3d_root data/cari4d-demo/meshes-sam3d \
    --sam3d_root sam-3d-objects \
    --blender_path /path/to/blender      # or --no_blender to write the OBJ with trimesh
```

On the cluster, `scripts/slurm_sam3d_recon.sh` is the drop-in for `slurm_hy3d_recon.sh`.
`--skip_sam3d` writes only the RGBA crop. The mesh is renormalised to the Hunyuan3D
convention (longest axis in `[-1, 1]`) so `tools/estimate_scale_video.py` sees no difference.

---

## Step 3: 2D Human Keypoints

Run 2D human body keypoint detection and pack the results into a pkl file.

### Output format

`<packed_root>/<seq>_GT-packed.pkl` (saved with `joblib.dump`) — a dict with the following keys:

| Key | Type | Shape | Description |
|---|---|---|---|
| `frames` | `list[str]` | `[N]` | 6-digit zero-padded frame indices, e.g. `['000000', '000001', ...]` |
| `joints2d` | `ndarray` | `(N, K, J, 3)` float64 | 2D keypoints per frame. `K` = number of views (1 for in-the-wild); `J` = 17 (COCO) or 25 (OpenPose); last dim = `(x, y, confidence)` |

The pipeline auto-detects the keypoint format from the `J` dimension — no configuration flag is needed.

### Option A: Sapiens (recommended)

[Sapiens](https://github.com/facebookresearch/sapiens) predicts COCO 17 keypoints and runs inside the `cari4d` conda env.

**Important**: You need to download the [joint regressor](https://github.com/hongsukchoi/Pose2Mesh_RELEASE/blob/master/data/COCO/J_regressor_coco.npy) to use COCO 17 keypoints, place it under folder `data/assets`. 

**Setup:**

```bash
# 1. Clone Sapiens into the project root
git clone https://github.com/facebookresearch/sapiens.git

# 2. Install dependencies (in the cari4d conda env)
conda activate cari4d
pip install mmcv-lite mmengine mmdet mmpretrain xtcocotools json_tricks munkres

# 3. Authenticate with HuggingFace for checkpoint access
#    Request access at https://huggingface.co/noahcao/sapiens-pose-coco, then:
huggingface-cli login --token $HF_TOKEN

# 4. Download the Sapiens 0.3b pose checkpoint
mkdir -p ~/sapiens_host/pose/checkpoints/sapiens_0.3b
huggingface-cli download noahcao/sapiens-pose-coco \
    sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796.pth \
    --local-dir ~/sapiens_host/pose/checkpoints
#    Alternatively, use the 0.6b model for higher accuracy (requires more VRAM):
#    huggingface-cli download noahcao/sapiens-pose-coco \
#        sapiens_0.6b/sapiens_0.6b_coco_best_coco_AP_812.pth \
#        --local-dir ~/sapiens_host/pose/checkpoints
```

**Example:**

```bash
python prep/run_sapiens_pose.py \
    --video data/cari4d-demo/wild/videos/<seq>.0.color.mp4 \
    --masks_root data/cari4d-demo/wild/masks \
    --packed_root data/cari4d-demo/wild/packed-coco
```

The script uses person masks to compute bounding boxes for top-down pose estimation. See `prep/run_sapiens_pose.py` for additional options (`--checkpoint`, `--batch_size`, `--device`).

### Option B: OpenPose

Run [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to detect Body 25 keypoints, and pack the results into the same pkl format above. See the OpenPose documentation for installation and usage.

---

## Step 4: Run the Pipeline

Run the full CARI4D pipeline. The video file should be an MP4 (`<seq>.0.color.mp4`) placed under `data/cari4d-demo/wild/videos/`.

```bash
bash scripts/demo-custom.sh data/cari4d-demo/wild/videos/<seq>.0.color.mp4
```

Update `packed_root` in the script to point to your keypoint output directory (e.g. `packed-coco` for Sapiens).
