# PushT BT v2：100,000更新の学習記録

実施：2026-09-08〜09。単一学習seed 3072。13:20頃に正常終了し、最終checkpointを保存。これは開始時の記録であり、再実行の指示ではありません。

最終lossは0.0862368、validation予測MSEは0.00441758。最終更新のLRは0です。[制御評価結果](PUSHT_CHECKPOINT_EVALUATION.ja.md)と[新規学習手順](../TRAINING.ja.md)は別文書です。定期監視は終了しています。

## 設定と保存物

run: `output/pusht/bt_spectral_v2_100k_s3072/`。学習出力・ログは`output/`以下へまとめ、全体をGit対象外にする。`*.log`もGit対象外。完了済みの今回のrunは旧`.cache/stable-wm/pusht/`へのシンボリックリンクで参照し、書込み中の実体や保存済みconfigは変更しない。初期値・console・source snapshot・完了済み短期診断にも`output/`からリンクを用意した。今後の新規runは`--output output/pusht/NEW_RUN`へ直接保存する。

短期診断からresumeせず、新規の共有初期値を別途生成した。後日Raw/TCを比較する場合は同じ初期値・データ順を使い、学習開始には別途指示を得る。以下は開始時の実コマンドで、旧パスを履歴として保持する。既存runへ重複実行しない。

```bash
.venv/bin/python mylewm/tools/create_shared_initialization.py --seed 3072 --output .cache/stable-wm/pusht/bt_spectral_v2_100k_shared_initialization_s3072.pt
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python mylewm/train_rbg.py train --mode bt --bt-depth 2 --bt-kappa .2 --bt-hidden 192 --steps 100000 --batch-size 128 --warmup-steps 500 --lr 5e-5 --min-lr 0 --workers 4 --save-every 5000 --diagnostics-every 1000 --seed 3072 --deterministic --initialization .cache/stable-wm/pusht/bt_spectral_v2_100k_shared_initialization_s3072.pt --output .cache/stable-wm/pusht/bt_spectral_v2_100k_s3072
```

上記trainerを実際にはsystemd user service `bt-pusht-spectral-v2-100k-s3072.service` で起動し、チャットのコマンド待機と分離した。consoleは同階層の `bt_spectral_v2_100k_s3072.console.log`。生存確認は `systemctl --user status bt-pusht-spectral-v2-100k-s3072.service`。同じコマンドを重複起動しない。自動再起動は設定していない。

1,000ステップごとにT診断、5,000ごとにvalidation・推論checkpoint・再開用checkpointを保存。100,000回目のLRは既存cosine仕様により0。推論モデル18,034,478、訓練専用T147,840パラメータ。12,800,000クリップ提示で、公式100エポック再現ではない。

開始時の未コミットコードも `bt_spectral_v2_100k_s3072.sources.tar.gz` に保存し、個別学習ソースのSHA256はconfigへ記録する。共有初期モデルstate hashは `1c1962d31c04bb65617d468686d5f23efe777058988c032408429d64dc02bc7d`。保存済みの学習ソースsnapshot・configを改変しない。
