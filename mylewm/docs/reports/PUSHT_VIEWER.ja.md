# PushT viewer：Goal付きMP4・保存整理・ブラウザ検証（2026-09-11）

動画単体でも判定対象が分かるよう、実行動画の右に評価時の固定Goal画像を入れました。Goal画像・状態のhash照合と保存状態からの成功再計算を通してから生成します。CEMや環境を再実行せず、元の15fps・50フレームを維持します。試行番号・現在step・位置／角度の判定数値も動画に含めます。灰色Tと青い操作点が対象で、背景の緑Tへの被覆率評価ではありません。

## 保存先

各評価の`viewer/index.html`を正式な入口にし、再生・ダウンロードに同じ`viewer/videos/trial_001.mp4`以降のファイルを使います。元の評価記録`env_N.mp4`は評価ディレクトリに1組だけ残します。assetsに元動画をコピーしません。

既存10k・20kの各50ケースを更新しました。動画生成時間は12.09秒／11.82秒でした。重複していた生成済みviewer・元動画コピーのMP4計500本、約19.8MBを整理しました。元の評価記録100本は前後SHA-256一致、Goal付き動画は各viewerに50本だけです。以前の`output/pusht/ui_bt_compiled_step{10000,20000}_seed42/index.html`には正式viewerへの移動ページだけ残します。

旧入口からの移動もURLの試行番号・フィルターを引き継ぎます。10k／20kのHTTP／fileの4通りで、旧入口の`#trial=27&filter=all`から正式viewerの27回目へ移動することを確認しました。

```text
output/pusht/eval_bt_compiled_step20000_seed42_cached/
  env_0.mp4 ...                 評価時の元記録
  results.txt.json / status.json
  viewer/
    index.html / report.json / results.txt
    assets/                    初期・Goal・判定フレームPNG
    videos/trial_001.mp4 ...    Goal付き動画（1回目から）
```

## Playwrightで再現した問題と修正

1. Range非対応のHTTPサーバーでは、動画全体がbuffer済みでもChromiumのseekable範囲が`[0,0]`となり、シークが先頭へ戻りました。HTTPでは現在の1本をBlobとして読み込み、シーク可能にしました。file URLでは直接再生します。
2. フィルター操作で先頭の試行を選び直し、再読み込みでも選択が1回目に戻っていました。27回目を選択→再読み込みで1回目になることをPlaywrightで再現しました。選択とフィルターをURLの`#trial=27&filter=all`に保持し、再読み込み・ブラウザの前後移動でも同じ試行を表示します。フィルター対象外でも選択を維持してその旨を表示し、不正な試行番号では明示エラーを出します。無関係な1回目へ置換しません。
3. 動画サイズを読み込むまで表示高さが変わる構成を、448:304の固定領域に変更しました。
4. 連続切替では前のfetchを中断し、遅れて届いた応答を世代番号で無効化します。Blob URLは切替・離脱時に解放し、戻る操作でページが復元された場合は再読込します。
5. 読み込み中・失敗を表示し、再生可能になるまで判定時点への移動を無効にします。

Playwright 1.63.0／Chromium 153.0.8010.12で、10k・20kのHTTPとfile URLをそれぞれ確認しました。**26合格・2スキップ、失敗・flakyなし**。スキップ2件はfile URLに適用しないHTTP遅延／404テストです。

再生、停止、判定数値、判定時点へのシーク、最終フレーム、絞り込み後の選択保持、再読み込み・履歴移動・不正URL、39回の連続前後切替、表示高さ、390px幅、全150アセットのHTTP参照（各評価）、遅延応答、404からの復帰、Blob解放を確認しました。追加で4・27・50回目を通常速度で終了まで再生し、4倍速で再再生しても試行・動画・Goal・選択表示が一致することを確認しました。ユーザー報告の「他の回の再生後」の自動切替そのものは再現せず、確認できた選択喪失経路を修正しています。Safari／Firefox／実スマートフォンは未確認です。

回帰テストは`mylewm/tests/browser/pusht_viewer.spec.cjs`。Playwrightは`/tmp/bt-viewer-playwright/`へ隔離導入し、Python学習環境を変更していません。ブラウザ結果は`output/pusht/viewer_debug_20260911/`に保存しました。

```bash
# Playwright導入例。既存のNodeプロジェクトや学習環境を変更しない。
npm install --prefix /tmp/bt-viewer-playwright @playwright/test@1.63.0
PLAYWRIGHT_BROWSERS_PATH=/tmp/bt-viewer-playwright/browsers \
  node /tmp/bt-viewer-playwright/node_modules/playwright/cli.js install chromium

# 保存済みviewerを直接開くテスト（HTTP遅延テストは対象外）。
PLAYWRIGHT_BROWSERS_PATH=/tmp/bt-viewer-playwright/browsers \
NODE_PATH=/tmp/bt-viewer-playwright/node_modules \
BT_VIEWER_URL="file://$PWD/output/pusht/eval_bt_compiled_step20000_seed42_cached/viewer/index.html" \
  node /tmp/bt-viewer-playwright/node_modules/playwright/cli.js test \
  --config mylewm/tests/browser/playwright.config.cjs
```

HTTPを確認する場合はローカルサーバーを用意し、`BT_VIEWER_URL`を対応するURLへ変更します。今回のRange非対応サーバーは`python -m http.server --bind 127.0.0.1 --directory output/pusht 18763`です。
