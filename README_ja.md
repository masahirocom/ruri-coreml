# ruri-coreml

[cl-nagoya/ruri-v3](https://huggingface.co/collections/cl-nagoya/ruri-japanese-general-text-embeddings)(日本語文埋め込みモデル)をiOS上でオンデバイス動作させるためにCore ML変換したものと、Apple標準搭載のオンデバイス埋め込みモデル(`NLContextualEmbedding`)と比較するiOSアプリです。

*(English README: [README.md](README.md))*

**必要環境:** Xcode 16以降、実機iPhone(シミュレータを推奨しない理由は`REPORT.md`参照)、[XcodeGen](https://github.com/yonaskolb/XcodeGen)、Python 3.10以降(自分でモデル変換する場合のみ)。

## 構成

- **[`RuriDemo/`](RuriDemo/)** — ruri-v3をCore ML経由で実行し、Appleの`NLContextualEmbedding`と同一クエリで並べて比較できるSwiftUI製iOSアプリ。実機上で直接ランキングを比較できます。
- **[`ruri_coreml/ruri_coreml_convert/`](ruri_coreml/ruri_coreml_convert/)** — ruri-v3のチェックポイントをCore MLに変換・量子化し、iOSアプリに同梱できる形式に整えるPythonパイプライン。
- **[`REPORT.md`](REPORT.md)** — 現代日本語・古典文語・英語にまたがる意味検索タスクで、ruri-v3とAppleのオンデバイス日本語/英語埋め込みモデルを比較した検証結果のレポート。

## 変換済みモデル

変換済みの `.mlpackage` はHugging Faceで公開しています。異なるサイズ・系列長・精度が必要な場合を除き、自分で変換する必要はありません。

- [masahiroid/ruri-v3-130m-coreml](https://huggingface.co/masahiroid/ruri-v3-130m-coreml)
- [masahiroid/ruri-v3-310m-coreml](https://huggingface.co/masahiroid/ruri-v3-310m-coreml)

各サイズとも **fp16版** と、そこから重みを量子化した **int8版** を系列長128/256/512で公開しています。iPhone 17 Pro実機(iOS 27)ではどちらもGPU/Neural Engine上で1文あたり約5msで動作します。int8版はサイズが半分でスコアもほぼ同じなので、モバイルアプリにはint8版を推奨します。

## iOSアプリの実行方法

```bash
cd RuriDemo
./scripts/download_model.sh   # Hugging Faceから130mのfp16版・int8版をRuriDemo/RuriDemo/Resourcesに取得
xcodegen generate
open RuriDemo.xcodeproj
```

Xcodeで自分の署名チームを設定し、必要に応じてダウンロードした `.mlpackage` をターゲットに追加して、実機で実行してください。詳細は [`RuriDemo/MANUAL.md`](RuriDemo/MANUAL.md) を参照してください。

## 自分でモデルを変換する場合

[`ruri_coreml/README.md`](ruri_coreml/README.md) を参照してください。

## Swift/iOS実装マニュアル

モデルのダウンロードと配置・読み込み・トークナイズ・推論は [`RuriDemo/MANUAL.md`](RuriDemo/MANUAL.md) にまとめています。

## ライセンス

ベースモデルの[cl-nagoya/ruri-v3](https://huggingface.co/collections/cl-nagoya/ruri-japanese-general-text-embeddings)と同じApache License 2.0です。本リポジトリはモデル形式の変換のみを行っており、ruri-v3自体の著作権・功績はオリジナルの開発チーム(名古屋大学、cl-nagoya)に帰属します。
