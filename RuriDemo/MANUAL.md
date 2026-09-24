# ruri-v3 Core ML モデル利用マニュアル(Swift / iOS・macOS)

このマニュアルは、`.mlpackage` 形式に変換した ruri-v3 モデルを Swift(iOS/macOS)アプリから利用する手順をまとめたものです。実装例は本リポジトリの [`RuriDemo`](.) アプリを参照してください。

## 1. 公開モデル

Hugging Face で次の2リポジトリを公開しています。各サイズとも **fp16版** と、そこから重みを int8 に量子化した **int8版** の2種類です(系列長 128 / 256 / 512 を用意)。

| リポジトリ | fp16 | int8(推奨) |
|---|---|---|
| [masahiroid/ruri-v3-130m-coreml](https://huggingface.co/masahiroid/ruri-v3-130m-coreml) | 265MB | 133MB |
| [masahiroid/ruri-v3-310m-coreml](https://huggingface.co/masahiroid/ruri-v3-310m-coreml) | 630MB | 316MB |

ファイル名は `ruri-v3-<サイズ>_seq<系列長>_<fp16|int8>.mlpackage` です(例: `ruri-v3-130m_seq128_int8.mlpackage`)。

iPhone 17 Pro(iOS 27)実機では、どちらも GPU/Neural Engine 上で動作し、130m の定常推論は1文あたり約5msでした。モバイルアプリには **130m の int8 版** を推奨します。

## 2. モデルのダウンロードと配置

モデル本体(重み)はサイズが大きいため Git リポジトリには含めていません。Hugging Face からダウンロードして `RuriDemo/RuriDemo/Resources/` に置いてください。

### スクリプトで取得する(RuriDemo 用)

```bash
cd RuriDemo
./scripts/download_model.sh
```

`ruri-v3-130m_seq128_fp16.mlpackage` と `ruri-v3-130m_seq128_int8.mlpackage` が `RuriDemo/RuriDemo/Resources/` に保存されます。別のサイズ・系列長が必要な場合は、スクリプト内の `REPO` と `MODEL_NAMES` を書き換えてください。

### 手動で取得する

`.mlpackage` はフォルダなので、中の3ファイルをそのままの階層で保存します。

```
ruri-v3-130m_seq128_int8.mlpackage/
├── Manifest.json
└── Data/com.apple.CoreML/
    ├── model.mlmodel
    └── weights/weight.bin
```

```bash
BASE=https://huggingface.co/masahiroid/ruri-v3-130m-coreml/resolve/main
M=ruri-v3-130m_seq128_int8.mlpackage
for f in Manifest.json Data/com.apple.CoreML/model.mlmodel Data/com.apple.CoreML/weights/weight.bin; do
  curl -fL --create-dirs -o "RuriDemo/RuriDemo/Resources/$M/$f" "$BASE/$M/$f"
done
```

`huggingface-cli download masahiroid/ruri-v3-130m-coreml --include "ruri-v3-130m_seq128_int8.mlpackage/*" --local-dir RuriDemo/RuriDemo/Resources` でも同じ結果になります。

### Xcode に追加する

1. ダウンロードした `.mlpackage` を Xcode の `Resources` グループにドラッグし、「Add to targets」でアプリのターゲットにチェックを入れる(ビルド時に自動で `.mlmodelc` にコンパイルされる)。
2. トークナイザフォルダ(`tokenizer.json` + `tokenizer_config.json`)は**フォルダ参照(青いフォルダ)**として追加する。通常のグループ参照だとファイルがバンドル直下に展開され、`Bundle.main.url(forResource: "tokenizer", withExtension: nil)` で見つからなくなる。トークナイザは `ruri_coreml_convert export-tokenizer` で生成できる(RuriDemo には同梱済み)。
3. Swift Package 依存関係に [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) を追加する(`Tokenizers` ライブラリを使用)。

RuriDemo では、`RuriModelConfiguration.swift` の `allKnownVariants` に並べたモデルがアプリ内のピッカーで切り替えられます。モデルを追加した場合はここに1行足してください。

## 3. モデルの読み込み

```swift
import CoreML

let modelConfiguration = MLModelConfiguration()
modelConfiguration.computeUnits = .all // CPU/GPU/Neural Engine を自動選択

let modelURL = Bundle.main.url(forResource: "ruri-v3-130m_seq128_int8", withExtension: "mlmodelc")!
let model = try MLModel(contentsOf: modelURL, configuration: modelConfiguration)
```

初回の推論だけ Neural Engine 向けの準備で時間がかかります(int8版で約2秒)。アプリ起動時に1回ダミー推論をしておくと、ユーザー操作時の待ちをなくせます。

## 4. トークナイザの読み込みと入力の構築

```swift
import Tokenizers

let tokenizerDir = Bundle.main.url(forResource: "tokenizer", withExtension: nil)!
let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)

let sequenceLength = 128 // モデルの系列長(ファイル名の seqXXX)に合わせる
let prefix = "検索文書: " // 1+3プレフィックススキーム。§7参照
let text = prefix + "瑠璃色は紫みを帯びた濃い青である。"

var tokenIDs = tokenizer.encode(text: text)
if tokenIDs.count > sequenceLength {
    tokenIDs = Array(tokenIDs.prefix(sequenceLength))
}
let paddingCount = sequenceLength - tokenIDs.count
let paddedIDs = tokenIDs + Array(repeating: 0, count: paddingCount)
let attentionMask = Array(repeating: 1, count: tokenIDs.count) + Array(repeating: 0, count: paddingCount)

let inputIDs = try MLMultiArray(shape: [1, NSNumber(value: sequenceLength)], dataType: .int32)
let attentionMaskArray = try MLMultiArray(shape: [1, NSNumber(value: sequenceLength)], dataType: .int32)
for i in 0..<sequenceLength {
    inputIDs[i] = NSNumber(value: paddedIDs[i])
    attentionMaskArray[i] = NSNumber(value: attentionMask[i])
}
```

## 5. 推論と出力の取得

```swift
let input = try MLDictionaryFeatureProvider(dictionary: [
    "input_ids": MLFeatureValue(multiArray: inputIDs),
    "attention_mask": MLFeatureValue(multiArray: attentionMaskArray),
])
let output = try model.prediction(from: input)
let embeddingArray = output.featureValue(for: "sentence_embedding")!.multiArrayValue!
```

fp16/int8 版の出力は `.float16` 型です。Swift の `Float16` 型として直接読み取ってください。

```swift
extension MLMultiArray {
    func asFloatVector() -> [Float] {
        guard dataType == .float16 else {
            return (0..<count).map { self[$0].floatValue }
        }
        let pointer = dataPointer.bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(pointer[$0]) }
    }
}
```

出力ベクトルはモデル内で mean pooling + L2 正規化済みなので、追加の後処理は不要です。2つのベクトルの類似度は内積(dot product)で計算できます(コサイン類似度と等価)。

## 6. シミュレータでの実行に関する注意

一部の macOS ベータ版では、Core ML のシミュレータ実行バックエンド(MPSGraph)にバグがあり、`computeUnits` の既定設定でモデルのロード・推論が失敗することがあります。シミュレータでのみ CPU 実行に固定すると回避できます。

```swift
#if targetEnvironment(simulator)
modelConfiguration.computeUnits = .cpuOnly
#else
modelConfiguration.computeUnits = .all
#endif
```

## 7. プレフィックススキーム

ruri-v3 は「1+3プレフィックススキーム」を採用しています。埋め込み対象のテキストの役割に応じて、以下のいずれかを先頭に付与してください。

| プレフィックス | 用途 |
|---|---|
| (空文字列) | 汎用的な意味的類似度 |
| `トピック: ` | 分類・クラスタリング |
| `検索クエリ: ` | 検索のクエリ側 |
| `検索文書: ` | 検索の対象文書側 |

## 8. 参考

- モデル変換パイプライン: [`ruri_coreml/ruri_coreml_convert/`](../ruri_coreml/ruri_coreml_convert/)
- 実装例(完全版): [`RuriDemo/RuriDemo/Embedding/`](RuriDemo/Embedding/)
- 詳細な検証結果: [`REPORT.md`](../REPORT.md)
