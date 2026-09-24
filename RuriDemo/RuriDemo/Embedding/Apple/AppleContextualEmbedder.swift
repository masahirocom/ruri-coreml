import NaturalLanguage

/// Wraps Apple's on-device NLContextualEmbedding (a BERT-like transformer
/// built into the OS since iOS 17) as a `SentenceEmbedder`, so it can be
/// compared against ruri-v3 through the same search code path.
///
/// NLContextualEmbedding returns one vector per token; this mean-pools and
/// L2-normalizes them, mirroring how ruri-v3's own pooling works, so the two
/// models' outputs are at least comparably *shaped* (their vector spaces are
/// still unrelated — only rankings within one model are meaningful).
actor AppleContextualEmbedder: SentenceEmbedder {
    private let model: NLContextualEmbedding
    private let language: NLLanguage

    static func load(
        language: NLLanguage,
        retryPolicy: ModelAssetLoadRetryPolicy = .standard
    ) async throws -> AppleContextualEmbedder {
        guard let model = NLContextualEmbedding(language: language) else {
            throw AppleEmbeddingError.modelUnavailable(language: language, underlying: nil)
        }

        if !model.hasAvailableAssets {
            let assetsResult = try await model.requestAssets()
            debugLog("[AppleContextualEmbedder] requestAssets(\(language.rawValue)) -> \(assetsResult)")
        }

        do {
            try await retryPolicy.run(
                { try model.load() },
                onAttemptFailure: { attempt, error in
                    debugLog("[AppleContextualEmbedder] load attempt \(attempt) failed for \(language.rawValue): \(error)")
                }
            )
        } catch {
            throw AppleEmbeddingError.modelUnavailable(language: language, underlying: error)
        }

        return AppleContextualEmbedder(model: model, language: language)
    }

    private init(model: NLContextualEmbedding, language: NLLanguage) {
        self.model = model
        self.language = language
    }

    func embed(_ text: String) throws -> [Float] {
        let (vector, milliseconds) = try measuringElapsedMilliseconds {
            try embedWithoutTiming(text)
        }
        debugLog("[AppleContextualEmbedder] embed(\(language.rawValue)) took \(String(format: "%.1f", milliseconds))ms for \(text.count) chars")
        return vector
    }

    private func embedWithoutTiming(_ text: String) throws -> [Float] {
        let result = try model.embeddingResult(for: text, language: language)

        var tokenVectors: [[Double]] = []
        result.enumerateTokenVectors(in: text.startIndex..<text.endIndex) { vector, _ in
            tokenVectors.append(vector)
            return true
        }
        guard !tokenVectors.isEmpty else { throw AppleEmbeddingError.noTokenVectorsProduced }

        let pooled = VectorMath.mean(of: tokenVectors).map(Float.init)
        return VectorMath.l2Normalized(pooled)
    }
}
