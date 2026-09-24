/// Turns a piece of text into a fixed-length vector suitable for comparing
/// against other text via `VectorMath.dotProduct`. Every embedder (ruri-v3,
/// Apple's NLContextualEmbedding, or any future addition) implements this
/// one method and nothing else — how the vector is produced is entirely the
/// implementer's concern.
protocol SentenceEmbedder: Sendable {
    /// Returns an L2-normalized embedding for `text`.
    func embed(_ text: String) async throws -> [Float]
}
