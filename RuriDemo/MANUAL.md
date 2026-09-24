# ruri-v3 Core ML モデル利用マニュアル(Swift / iOS・macOS)

このマニュアルは、`.mlpackage` 形式に変換した ruri-v3 モデルを、Swift(iOS/macOS)アプリから利用する手順をまとめたものです。実装例は本リポジトリの [`RuriDemo`](.) アプリを参照してください。

## 1. 必要なもの

- 変換済みモデル: `ruri-v3-130m_seq128_fp32.mlpackage`(推奨。理由は §5 参照)
- トークナイザフォルダ: `tokenizer.json` + `tokenizer_config.json`(`ruri_coreml_convert export-tokenizer` で生成)
- [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) の `Tokenizers` ライブラリ(Swift Package Manager)

## 2. Xcodeプロジェクトへの組み込み

1. `.mlpackage` フォルダをXcodeプロジェクトの `sources` に追加する(ビルド時に自動で `.mlmodelc` にコンパイルされる)。
2. トークナイザフォルダは**フォルダ参照(青いフォルダ)**として追加する。通常のグループ参照だとファイルがバンドル直下にフラット展開され、`Bundle.main.url(forResource: "tokenizer", withExtension: nil)` で見つからなくなる。xcodegenを使う場合は `sources` エントリに `type: folder` を明示する。
3. Package.swiftまたはXcodeのSwift Package依存関係に `https://github.com/huggingface/swift-transformers` を追加する。

## 3. モデルの読み込み

```swift
import CoreML

let modelConfiguration = MLModelConfiguration()
modelConfiguration.computeUnits = .all // 実機推奨。シミュレータでは既知の問題があるため .cpuOnly を検討(§6)

let modelURL = Bundle.main.url(forResource: "ruri-v3-130m_seq128_fp32", withExtension: "mlmodelc")!
let model = try MLModel(contentsOf: modelURL, configuration: modelConfiguration)
```

## 4. トークナイザの読み込みと入力の構築

```swift
import Tokenizers

let tokenizerDir = Bundle.main.url(forResource: "tokenizer", withExtension: nil)!
let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)

let sequenceLength = 128
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

**重要:** `embeddingArray[i].floatValue` という `NSNumber` 経由の読み取り方法は、出力dtypeが `.float16` の場合に**実機でNaNを返す既知のバグ**があります(シミュレータでは問題なく動作するため発見が遅れやすい)。float16出力を扱う場合は、必ずSwiftネイティブの `Float16` 型で直接読み取ってください。

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

## 6. fp16モデルは実機で使わない

`fp16` 精度で変換したモデルは、§5の読み取り修正を行ってもなお、**実機のCPU上で計算そのものがNaNを返す**ことが確認されています(RoPEの高周波成分やLayerNormでのオーバーフローが疑われる)。シミュレータでは内部的にfloat32へ昇格して計算されるためこの問題は再現しません。**実機で使うモデルは必ず `fp32` 精度で変換してください。**

## 7. シミュレータでの実行に関する注意

一部のmacOSベータ版では、CoreMLのシミュレータ実行バックエンド(MPSGraph)にバグがあり、`computeUnits` のデフォルト設定でモデルロード・推論が失敗することがあります。次のように、シミュレータでのみCPU実行に固定する回避策が有効です。

```swift
#if targetEnvironment(simulator)
modelConfiguration.computeUnits = .cpuOnly
#else
modelConfiguration.computeUnits = .all
#endif
```

## 8. プレフィックススキーム

ruri-v3は「1+3プレフィックススキーム」を採用しています。埋め込み対象のテキストの役割に応じて、以下のいずれかを先頭に付与してください。

| プレフィックス | 用途 |
|---|---|
| (空文字列) | 汎用的な意味的類似度 |
| `トピック: ` | 分類・クラスタリング |
| `検索クエリ: ` | 検索のクエリ側 |
| `検索文書: ` | 検索の対象文書側 |

## 9. 参考

- モデル変換パイプライン: [`ruri_coreml/ruri_coreml_convert/`](../ruri_coreml/ruri_coreml_convert/)
- 実装例(完全版): [`RuriDemo/RuriDemo/Embedding/`](RuriDemo/Embedding/)
- 詳細な検証結果: [`REPORT.md`](../REPORT.md)
