# Run CARI4D on custom videos.
# Please follow ./docs/custom_videos.md to prepare data before running this script.
# Required data before running: 1). Object mesh. 2). Masks of human and object. 3). openpose detections of the human.
#
# Every path below can be overridden from the environment, so one script serves
# sequences whose preprocessing landed in different places:
#
#   MASKS_ROOT=data/cari4d-demo/wild/masks \
#   PACKED_ROOT=data/cari4d-demo/wild/packed-coco \
#   HY3D_ROOT=data/cari4d-demo/meshes \
#       bash scripts/demo-custom.sh data/cari4d-demo/wild/videos/<seq>.0.color.mp4
#
# The defaults are the released videogen demo's layout, unchanged.

video=$1
video_prefix=$(basename "$video" | cut -d. -f1)
video_dir=$(dirname "$video")
echo $video_prefix

# Paths that store preprocessed data:
masks_root="${MASKS_ROOT:-data/cari4d-demo/videogen/masks/}" # store the masks of human and object.
packed_root="${PACKED_ROOT:-data/cari4d-demo/videogen/packed/}" # store the openpose detections for each frame.
hy3d_root="${HY3D_ROOT:-data/cari4d-demo/videogen/meshes}" # store the reconstructed object mesh in normalized scale.

# Paths for intermediate results:
nlf_path="${NLF_PATH:-data/cari4d-demo/videogen/nlf}"
fp_root="${FP_ROOT:-data/cari4d-demo/videogen/fp-hy3d3-track}"
coconet_out="${COCONET_OUT:-output/coconet}"

# The trained checkpoint Step 6 loads, and the directory Step 7 reads its output
# from. Kept together because they must agree: opt_refineout reads
# <coconet_out>/<exp_name>+step<N><identifier>/<seq>.pth, which run_horefine
# writes from cfg.exp_name and the checkpoint it found.
exp_name="${EXP_NAME:-cari4d-release}"
exp_step="${EXP_STEP:-step031397}"
identifier="${IDENTIFIER:-_demo}"

# First frame FoundationPose tracks, in the video's time units. BEHAVE videos
# carry real timestamps and 3.0 skips the setup seconds; a wild video has no
# time files, so times are frame indices and 3.0 silently starts at frame 3.
# That matters because register() runs on whichever frame is first: on the
# egoexo4d basketball, frames 4-8 carry depth outliers of 23-27m against an
# apparent-size distance of 4.5-7.4m, so tracking initialised on a frame whose
# depth was then clipped away and died with 'valid is empty'. Frame 0 reads
# 6.48m there, agreeing with apparent size to 12%.
tstart="${TSTART:-3.0}"

# Depth beyond this many metres is discarded before FoundationPose tracking.
# The 8m default matches BEHAVE's indoor capture volume; a scene shot at
# distance needs more, or there is no valid depth inside the object mask and
# tracking cannot initialise.
zfar="${ZFAR:-8.0}"

set -e

echo "masks_root=${masks_root}"
echo "packed_root=${packed_root}"
echo "hy3d_root=${hy3d_root}"
echo "nlf_path=${nlf_path}  fp_root=${fp_root}  coconet_out=${coconet_out}"
echo "exp=${exp_name}+${exp_step}${identifier}  zfar=${zfar}  tstart=${tstart}"

for required in "$video" "$masks_root" "$packed_root" "$hy3d_root"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: missing required input: $required" >&2
        exit 1
    fi
done

# Step 1: run Unidepth estimation
python prep/unidepth_behave.py --wild_video --video ${video} -o ${video_dir}

# Step 2: run NLF
python prep/run_nlf_sepK.py -o ${nlf_path} --masks_root ${masks_root} --video ${video} --wild_video

# Steps 3 and 4 both need an explicit -o. Neither writes anything there --
# fit_smplh_global derives its output from --nlf_path, align_monod2hum from the
# packed file -- but both inherit BaseBehaveVideoData's parser, whose
# -o/--outpath defaults to the original author's home
# (/home/xianghuix/datasets/behave/fp, behave_video.py:229) and gets mkdir'd
# during setup. Without this they die on PermissionError before doing any work.

# Step 3: run SMPLH fitting to get globally consistent human pose and translation
python prep/fit_smplh_global.py --wild_video --video ${video} --packed_root ${packed_root} --masks_root ${masks_root} \
    --nlf_path=${nlf_path} -o ${nlf_path}-opt

# Step 4: align Unidepth to GENMO human
python prep/align_monod2hum.py --wild_video --nlf_path ${nlf_path}-opt \
--masks_root ${masks_root} \
--video ${video} -o ${nlf_path}-opt

# Update the video path, pointing to the new video with aligned depth.
video=${video_dir}-aligned/${video_prefix}.0.color.mp4

# Step 5.1: estimate metric scale of the object
python tools/estimate_scale_video.py --wild_video --video ${video} --masks_root ${masks_root} --hy3d_root ${hy3d_root} -o ${hy3d_root}-metric


# Step 5.2: run FP in tracking mode
python prep/fp_hy3d_track.py --viz_path x --wild_video --kid 0 \
--masks_root ${masks_root} --hy3d_root=${hy3d_root}-metric \
--video ${video} -o ${fp_root} --zfar ${zfar} -tstart ${tstart}

# Step 6: run CoCoNet to refine human + object
python run_horefine.py config=learning/configs/cari4d-release.yml split_file=splits/demo-behave.json \
use_sel_view=True render_video=True identifier=${identifier} use_intermediate=False data_name=test-only \
hy3d_meshes_root=${hy3d_root}-metric \
masks_root=${masks_root} \
fp_root=${fp_root} \
nlf_root=${nlf_path}-opt \
video=${video}  cam_id=0 wild_video=True \
outpath=${coconet_out}

# Step 7: run joint optimization
python learning/training/opt_refineout.py num_steps=3000 w_acc_v=600 w_contact=300  save_name=optv2 batch_size=64 opt_rot=True \
opt_trans=True w_temp=1000 w_sil=0.002 w_contact=200.0 w_pen=2.0 w_j2d=0.006 opt_smpl_trans=False opt_betas=False  \
pth_file=${coconet_out}/${exp_name}+${exp_step}${identifier}/${video_prefix}.pth  wild_video=True use_input=True \
video_root=$(dirname "$video") \
packed_root=${packed_root} \
masks_root=${masks_root}  \
hy3d_meshes_root=${hy3d_root}-metric outpath=output/opt
# Note: reduce batch_size if encounter GPU OOM.
