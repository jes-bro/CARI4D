# Per-machine settings for the reconstruction pipeline. Copy, edit, source:
#
#   cp scripts/recon_env.example.sh ~/recon_env.sh
#   # edit it
#   source ~/recon_env.sh
#
# Source it once per shell, before running any recon_* script. Everything here
# is a default the scripts already fall back to, so an empty file works on the
# machine this was developed on -- you only set what differs.
#
# Put it OUTSIDE the repo (or leave the copy untracked): it is per-person, and
# committing one person's cluster into the shared history is how the paths got
# hardcoded in the first place.

# --- where the code is -------------------------------------------------------
# Not usually needed. The scripts take it from the directory sbatch ran in,
# which the drivers guarantee is the repo root. Set it only if you submit jobs
# from somewhere other than a checkout.
# export REPO=/path/to/CARI4D

# --- where the footage is ----------------------------------------------------
# The EgoExo4D mirror: <TAKES_ROOT>/<take_name>/frame_aligned_videos/...
export TAKES_ROOT=/vision/group/egoexo4d/takes

# --- where the outputs go ----------------------------------------------------
# One directory per clip lands here. Defaults to <repo>/work. Point it at
# scratch or project space if your checkout is on a small quota -- a clip's
# masks alone run to hundreds of MB.
# export WORK_ROOT=/path/with/room/work

# --- conda envs --------------------------------------------------------------
# Three, because they need incompatible torch versions. SAM3 needs Python 3.12+
# and torch 2.7+; hy3d needs Hunyuan3D-2 plus Blender; everything else is the
# main env.
export CARI4D_ENV=newcari4d
export SAM3_ENV=sam3
export HY3D_ENV=hy3d

# --- model weights and caches ------------------------------------------------
# Caches must NOT sit on a quota'd home directory: these download tens of GB and
# JIT-build CUDA extensions, and a full filesystem makes them stall rather than
# fail. CACHE_ROOT covers HF_HOME, TORCH_HOME, TORCH_EXTENSIONS_DIR and friends.
export CACHE_ROOT=/simurgh2/projects/ret-hoi
export CHECKPOINT=/simurgh2/projects/ret-hoi/sapiens_ckpts/sapiens_host/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796.pth

# --- Hugging Face token ------------------------------------------------------
# SAM3's weights are gated. Authenticate ONCE on the login node so the jobs
# never need to log in:
#   HF_HOME=$CACHE_ROOT/hf_cache huggingface-cli login --token <tok>
# That writes $HF_HOME/token, which the jobs read. Exporting HF_TOKEN in the
# submitting shell works too -- sbatch carries it through.
# export HF_TOKEN=

# --- cluster -----------------------------------------------------------------
# Nodes to keep away from. A GPU with failing memory reports "uncorrectable ECC
# error" and kills whatever lands on it, which reads like a pipeline bug.
# export EXCLUDE_NODES=

# Account, partition and QOS. The slurm_*.sh files carry #SBATCH lines naming
# this cluster's, but you do NOT need to edit them: sbatch reads these three
# variables and they take precedence over the directives inside a job script.
# Set them to your own and every stage follows.
# export SBATCH_ACCOUNT=your_account
# export SBATCH_PARTITION=your_partition
# export SBATCH_QOS=normal

# Job mail. This one has no Slurm environment variable, so the drivers pass it
# on the sbatch command line. Left unset, the #SBATCH --mail-user literal in
# each job script applies -- which means mail about YOUR jobs goes to the
# address this repo was developed with, so set it.
# export MAIL_USER=you@example.edu

# --- NOT set here: things that must exist in the checkout --------------------
# These are files and clones, not paths, so there is nothing to point at --
# they have to be present. See README/docs and CLAUDE.md:
#
#   sam3/                     clone of facebookresearch/sam3, pip install -e
#   Hunyuan3D-2/              clone, plus an extracted blender-*/ inside it
#   unidepth/                 clone of UniDepth
#   VolumetricSMPL/           clone, patched with scripts/volumetric_smplh.patch
#   weights/                  NLF + FoundationPose weights
#   experiments/cari4d-release/step031397.pth    the trained CoCoNet checkpoint
#   data/smpl/smplh/SMPLH_{male,female}.pkl      from the MANO project page
#   data/smpl/kid_template.npy                   from AGORA
#
# The SMPL-H models are not redistributable and must be downloaded per person.
#
# NOT needed, despite appearances: the BEHAVE dataset. behave_data/const.py
# hardcodes BEHAVE_ROOT to the original author's home, but every path that
# reads it is dead for wild sequences -- opt_refineout hardcodes use_hy3d=True
# above it, and video_data guards it with `if not cfg.wild_video`.
