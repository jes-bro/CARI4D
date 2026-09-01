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

# --- cluster -----------------------------------------------------------------
# Nodes to keep away from. A GPU with failing memory reports "uncorrectable ECC
# error" and kills whatever lands on it, which reads like a pipeline bug.
# export EXCLUDE_NODES=

# The #SBATCH --account, --partition and --mail-user lines are still literals
# inside scripts/slurm_*.sh. On a different cluster those need editing there;
# there is no environment override for them yet.
