/// Ranks a set of documents against a query by embedding similarity, using
/// whichever `SentenceEmbedder` it's given. This is the one place that
/// contains the actual search algorithm — every comparison target (ruri-v3,
/// Apple's Japanese model, Apple's English model, ...) reuses it instead of
/// re-implementing its own copy.
struct SimilaritySearchService {
    /// Some models (ruri-v3) want a different prefix applied to the query
    /// than to documents; others (Apple's) want none. Callers express that
    /// by passing two embedders — they may be the same instance, or two
    /// `PrefixedSentenceEmbedder`s wrapping one shared model.
    func rank(
        query: String,
        against documents: [Document],
        queryEmbedder: any SentenceEmbedder,
        documentEmbedder: any SentenceEmbedder
    ) async throws -> [SimilarityResult] {
        let queryVector = try await queryEmbedder.embed(query)

        var results: [SimilarityResult] = []
        results.reserveCapacity(documents.count)
        for document in documents {
            let documentVector = try await documentEmbedder.embed(document.text)
            let score = VectorMath.dotProduct(queryVector, documentVector)
            results.append(SimilarityResult(document: document, score: score))
        }

        return results.sorted { $0.score > $1.score }
    }
}
