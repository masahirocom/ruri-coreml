/// Everything the ruri-v3 CoreML embedder needs to know about one specific
/// exported model variant. Bundling this in one place means adding a new
/// variant (different size, sequence length, or precision) touches exactly
/// this file — see `allKnownVariants`.
struct RuriModelConfiguration: Identifiable, Hashable {
    /// Basename of the compiled `.mlmodelc` resource in the app bundle
    /// (without extension). Must match the `.mlpackage` added to the
    /// Xcode target — see RuriDemo/Resources. Doubles as this
    /// configuration's stable identity.
    let compiledModelResourceName: String
    var id: String { compiledModelResourceName }

    /// Shown in the in-app model picker.
    let displayName: String

    /// The fixed token sequence length the model was exported with. Inputs
    /// longer than this are truncated; shorter ones are padded. This is a
    /// property of the exported CoreML graph, not a tunable knob — changing
    /// it here without re-exporting the model will produce wrong shapes.
    let sequenceLength: Int

    /// Name of the folder resource (not a single file) bundled with the
    /// tokenizer's `tokenizer.json` / `tokenizer_config.json`.
    let tokenizerResourceDirectoryName: String

    /// Name of the model's output feature, as declared during CoreML export.
    let outputFeatureName: String

    /// Requires `ruri-v3-130m_seq128_fp16.mlpackage` in Resources
    /// (fetch it with `scripts/download_model.sh`).
    static let ruriV3_130M_Seq128_fp16 = RuriModelConfiguration(
        compiledModelResourceName: "ruri-v3-130m_seq128_fp16",
        displayName: "ruri-v3-130m",
        sequenceLength: 128,
        tokenizerResourceDirectoryName: "tokenizer",
        outputFeatureName: "sentence_embedding"
    )

    /// Requires `ruri-v3-130m_seq128_int8.mlpackage` in Resources
    /// (fetch it with `scripts/download_model.sh`).
    static let ruriV3_130M_Seq128_int8 = RuriModelConfiguration(
        compiledModelResourceName: "ruri-v3-130m_seq128_int8",
        displayName: "ruri-v3-130m 量子化(int8)",
        sequenceLength: 128,
        tokenizerResourceDirectoryName: "tokenizer",
        outputFeatureName: "sentence_embedding"
    )

    /// Used when nothing else has been selected.
    static let `default` = ruriV3_130M_Seq128_int8

    /// Every variant this app knows how to load. Powers the in-app model
    /// picker — add a bundled `.mlpackage` here (and to Xcode's Resources)
    /// to make it selectable without touching any other file.
    static let allKnownVariants: [RuriModelConfiguration] = [
        .ruriV3_130M_Seq128_int8,
        .ruriV3_130M_Seq128_fp16,
    ]
}
