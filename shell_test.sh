set -x
MODEL=$1

CUDA_VISIBLE_DEVICES=1 python test.py --load_model ${MODEL}
