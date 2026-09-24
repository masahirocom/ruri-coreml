import CoreML
import Tokenizers

/// Runs ruri-v3 (a Japanese ModernBERT-based text embedding model) on-device
/// via CoreML. Produces a mean-pooled, L2-normalized sentence embedding for
/// whatever text it's given — callers decide whether/how to prefix that text
/// (see `RuriPromptPrefix`, `PrefixedSentenceEmbedder`).
actor RuriCoreMLEmbedder: SentenceEmbedder {
    private let model: MLModel
    private let tokenizer: any Tokenizer
    private let configuration: RuriModelConfiguration

    static func load(
        configuration: RuriModelConfiguration = .ruriV3_130M_Seq128,
        computeUnits: ComputeUnitsPolicy = .default
    ) async throws -> RuriCoreMLEmbedder {
        guard let modelURL = Bundle.main.url(
            forResource: configuration.compiledModelResourceName,
            withExtension: "mlmodelc"
        ) else {
            throw RuriEmbeddingError.modelResourceNotFound(configuration.compiledModelResourceName)
        }

        let modelConfiguration = MLModelConfiguration()
        modelConfiguration.computeUnits = computeUnits.resolved()
        let model = try MLModel(contentsOf: modelURL, configuration: modelConfiguration)

        guard let tokenizerDirectoryURL = Bundle.main.url(
            forResource: configuration.tokenizerResourceDirectoryName,
            withExtension: nil
        ) else {
            throw RuriEmbeddingError.tokenizerResourceNotFound(configuration.tokenizerResourceDirectoryName)
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectoryURL)

        return RuriCoreMLEmbedder(model: model, tokenizer: tokenizer, configuration: configuration)
    }

    private init(model: MLModel, tokenizer: any Tokenizer, configuration: RuriModelConfiguration) {
        self.model = model
        self.tokenizer = tokenizer
        self.configuration = configuration
    }

    func embed(_ text: String) throws -> [Float] {
        let (vector, totalMilliseconds) = try measuringElapsedMilliseconds {
            try embedWithoutTiming(text)
        }
        debugLog("[RuriCoreMLEmbedder] embed took \(String(format: "%.1f", totalMilliseconds))ms total (tokenize + predict) for \(text.count) chars")
        return vector
    }

    private func embedWithoutTiming(_ text: String) throws -> [Float] {
        let encoding = RuriTokenEncoding(
            tokenIDs: tokenizer.encode(text: text),
            sequenceLength: configuration.sequenceLength
        )
        let input = try RuriModelInputBuilder.makeFeatureProvider(
            for: encoding,
            sequenceLength: configuration.sequenceLength
        )

        let (output, predictMilliseconds) = try measuringElapsedMilliseconds {
            try model.prediction(from: input)
        }
        debugLog("[RuriCoreMLEmbedder] model.prediction took \(String(format: "%.1f", predictMilliseconds))ms")

        guard let embeddingArray = output.featureValue(for: configuration.outputFeatureName)?.multiArrayValue else {
            throw RuriEmbeddingError.missingOutputFeature(configuration.outputFeatureName)
        }

        // The exported graph already mean-pools and L2-normalizes internally
        // (see the conversion pipeline), so no further math is needed here.
        return embeddingArray.asFloatVector()
    }
}
