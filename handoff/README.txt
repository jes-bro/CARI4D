CARI4D handoff archives
=======================

One tar per subject, grouped by scenario. Paths inside each archive are
relative to an EgoExo4D takes/ root, so:

    tar -xf expert387_takes.tar -C /your/egoexo4d/takes

puts every file where the pipeline scripts look for it. Each archive holds,
per take: the 4K exo videos, the whole frame_aligned_videos/downscaled/448
directory, and trajectory/gopro_calibs.csv.

MANIFEST.tsv lists every archive with its subject, take count, duration,
proficiency tier, gender, university, captures, cameras and tasks.

THINGS THAT WILL BITE YOU

Cameras are per capture, not per university. Some captures have five exo
cameras, some are missing cam02 entirely. scripts/recon_common.sh reads each
take's list off disk; do not copy an ALL_CAMS value between captures.

The pipeline camera comes from each take's own best_exo now, not a cam04
default. Confirm it against the stage-1 mask overlay before committing a
capture to a view.

The SAM3 prompts default to basketball. A pan, a soccer ball and a CPR
manikin all need OBJECT_PROMPT changed, and leaving the default in place is
how a sequence comes back with no masks at all.

REINIT_EVERY is derived from the object's symmetry. A ball gets
re-registration and cannot drift; a pan and a manikin have observable
orientation and must not be forced to 1, or re-registration will spin them.

CPR carries no proficiency tier. The Health scenario has no proficiency
annotations at all -- that is the dataset, not an omission.
