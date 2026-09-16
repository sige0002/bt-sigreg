# マシン固有設定の扱い

共通手順ではリポジトリ基準の変数を使います。実マシンのホーム、ディスクのマウント先、service名は各実験レポート・起動記録で確認します。

| 設定 | 扱い |
|---|---|
| データの保存ディスク | `PUSHT_DATA_ROOT`／`LIBERO_DATA_ROOT`を実行前に指定。manifest作成後に移動・改変しない |
| GPUの割当 | 併走時の資源と`CUDA_VISIBLE_DEVICES`を実行環境で決め、記録する |
| GB10のメモリ対策 | [GB10の対照試験](../reports/GB10_MEMORY_CONTROLS.ja.md)の対象環境だけの証拠。別GPUへ自動適用しない |
| PushTの`--gb10-cache-workaround` | 評価用の個別対策。共通評価の必須フラグではない |
| LIBEROの描画runtime | `LIBERO_ROOT`、`LIBERO_MUJOCO_PATH`、`LIBERO_OSMESA_DIR`、`LIBERO_CONFIG_PATH`を指定し、画像監査で確認 |

過去の「このPC」は、その記録時点の環境です。現在のEC2の保存先・GPU・導入済み依存の証拠にはしません。別マシンへ移す場合も、過去のmanifestやconfigを書き換えて同一runに見せません。
