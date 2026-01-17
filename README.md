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

# いくつかの注意事項
valid_feature_stats.py で保存したラベルと weighted_valid_feature_stats.py で取り出すターゲットラベルは、Biwi Kinect では同じ順序で並んでいます。理由は次のとおりです。

ラベルの並びは常に config["dataset"]["config"]["target"] の順番で決まります（例: ["yaw", "roll", "pitch"]）。
valid_feature_stats.py は calc.compute_stats() で取得した labels をそのまま feat_labels = torch.cat([features, labels], dim=1) として保存します。この labels は BiwiKinect の __getitem__ が target リスト順に組み立てたものです。
weighted_valid_feature_stats.py 側も同じ target_keys = config["dataset"]["config"]["target"] を読み、feat_labels を末尾から label_dim 個切り出して src_label にし、ターゲット側も metadata から for k in target_keys で並べています。
したがって、同じ config を使っている限り、yaw が yaw に、roll が roll に対応し、順番が入れ替わることはありません（target の順序を変える／異なる config で作った特徴ファイルを読む場合は要注意）。

やるべきこと
- [ ] UMAPによるアライメントの可視化
- [ ] UTKFace-Cにおいて非定常分布での平均，分散がどのように偏っているかをプロット
- [ ] 性能比較を棒グラフで作成（UTKFace, 4Seasons）＋　定常分布下でのSSAの棒グラフも追加
- [ ] EMAパラメータのalbation studyのプロット（これは散布図）
- [x] 実験セクションの流れを決める

