import NaturalLanguage

/// Knows how to turn a `ComparisonTargetID` into a ready-to-search
/// `LoadedComparisonTarget`. This is the one place that decides which
/// embedder backs which target and which prefixing it needs — adding a new
/// comparison target means adding one case here (and to
/// `ComparisonTargetID`), not touching the view model or search code.
enum ComparisonTargetLoader {
    static func load(_ id: ComparisonTargetID) async throws -> LoadedComparisonTarget {
        switch id {
        case .ruri:
            return try await loadRuri(id: id)
        case .appleJapanese:
            return try await loadApple(id: id, language: .japanese, query: MoonNostalgiaDataset.queryJapanese)
        case .appleEnglish:
            return try await loadApple(id: id, language: .english, query: MoonNostalgiaDataset.queryEnglish)
        }
    }

    private static func loadRuri(id: ComparisonTargetID) async throws -> LoadedComparisonTarget {
        let model = try await RuriCoreMLEmbedder.load()
        return LoadedComparisonTarget(
            id: id,
            query: MoonNostalgiaDataset.queryJapanese,
            documents: MoonNostalgiaDataset.documents,
            queryEmbedder: PrefixedSentenceEmbedder(wrapping: model, prefix: .searchQuery),
            documentEmbedder: PrefixedSentenceEmbedder(wrapping: model, prefix: .searchDocument)
        )
    }

    private static func loadApple(
        id: ComparisonTargetID,
        language: NLLanguage,
        query: String
    ) async throws -> LoadedComparisonTarget {
        let model = try await AppleContextualEmbedder.load(language: language)
        return LoadedComparisonTarget(
            id: id,
            query: query,
            documents: MoonNostalgiaDataset.documents(in: language),
            queryEmbedder: model,
            documentEmbedder: model
        )
    }
}
