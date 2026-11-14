# Site-specific settings for run_pipeline_tomo.sh / run_pipeline.sh.
#
#   cp config.example.sh config.local.sh   # config.local.sh is git-ignored
#   $EDITOR config.local.sh
#
# The ':=' form means a value given on the command line still wins, e.g.
#   SRC_H5=/other/rec.h5 bash run_pipeline_tomo.sh

: "${SRC_H5:=/path/to/rec.h5}"            # input HDF5 reconstruction
: "${OUT_BASE:=/path/to/output}"          # rec.zarr, renders/, logs/ go here
: "${ENV_PY:=$HOME/miniconda3/envs/zarr3_render/bin/python}"
: "${PIXEL_SIZE:=1.0}"                    # voxel size in micrometers
