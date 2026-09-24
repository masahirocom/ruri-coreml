import NaturalLanguage

enum AppleEmbeddingError: Error, LocalizedError {
    case modelUnavailable(language: NLLanguage, underlying: Error?)
    case noTokenVectorsProduced

    var errorDescription: String? {
        switch self {
        case let .modelUnavailable(language, underlying):
            let reason = underlying.map { ": \($0.localizedDescription)" } ?? ""
            return "Apple's NLContextualEmbedding is unavailable for \(language.rawValue)\(reason)."
        case .noTokenVectorsProduced:
            return "NLContextualEmbedding produced no token vectors for the given text."
        }
    }
}
