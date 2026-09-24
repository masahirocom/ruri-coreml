/// Wraps any `SentenceEmbedder` so every call is prefixed before delegating.
/// This lets one loaded ruri-v3 model serve as two logical embedders — a
/// "query" one and a "document" one — without loading the model twice.
struct PrefixedSentenceEmbedder: SentenceEmbedder {
    private let base: any SentenceEmbedder
    private let prefix: RuriPromptPrefix

    init(wrapping base: any SentenceEmbedder, prefix: RuriPromptPrefix) {
        self.base = base
        self.prefix = prefix
    }

    func embed(_ text: String) async throws -> [Float] {
        try await base.embed(prefix.applied(to: text))
    }
}
