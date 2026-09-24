/// Everything the ruri-v3 CoreML embedder needs to know about the specific
/// model variant it was built against. Bundling this in one place means
/// swapping in a different exported model (a different size or sequence
/// length) touches exactly this file.
struct RuriModelConfiguration {
    /// Basename of the compiled `.mlmodelc` resource in the app bundle
    /// (without extension). Must match the `.mlpackage` added to the
    /// Xcode target — see RuriDemo/Resources.
    let compiledModelResourceName: String

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

    static let ruriV3_130M_Seq128 = RuriModelConfiguration(
        compiledModelResourceName: "ruri-v3-130m_seq128_fp32",
        sequenceLength: 128,
        tokenizerResourceDirectoryName: "tokenizer",
        outputFeatureName: "sentence_embedding"
    )
}
