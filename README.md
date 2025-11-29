コマンド一覧
# 端末A（アーキA用・GPU0を占有）
CUDA_VISIBLE_DEVICES=0 \
python train_source.py -c configs/archA.yaml -o outputs/archA

# 端末B（アーキB用・GPU1を占有）
CUDA_VISIBLE_DEVICES=1 \
python train_source.py -c configs/archB.yaml -o outputs/archB

python feature_stats.py -c configs/feature_stats/utkface-Res50-BN.yaml -o models --validation --save_feature

# imagenet_cライブラリの修正
.venv/lib/python3.12/site-packages/imagenet_c/corruptions.py
multichannel -> channel_axis