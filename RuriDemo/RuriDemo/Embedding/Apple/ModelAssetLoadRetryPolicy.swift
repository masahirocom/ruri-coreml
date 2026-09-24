import Foundation

/// On first use, Apple downloads and compiles NLContextualEmbedding's model
/// assets on demand. Immediately after `requestAssets()` returns, `load()`
/// can still fail transiently (e.g. "requires compilation") while that
/// background compilation finishes — so loading is retried a bounded number
/// of times with a fixed delay rather than failing on the first attempt.
struct ModelAssetLoadRetryPolicy {
    let maxAttempts: Int
    let delayBetweenAttempts: Duration

    static let standard = ModelAssetLoadRetryPolicy(maxAttempts: 5, delayBetweenAttempts: .seconds(3))

    /// Runs `operation`, retrying on failure up to `maxAttempts` times with
    /// `delayBetweenAttempts` between tries. Rethrows the last failure if
    /// every attempt fails.
    func run<T>(_ operation: () throws -> T, onAttemptFailure: (Int, Error) -> Void = { _, _ in }) async throws -> T {
        var lastError: Error?
        for attempt in 1...maxAttempts {
            do {
                return try operation()
            } catch {
                lastError = error
                onAttemptFailure(attempt, error)
                if attempt < maxAttempts {
                    try? await Task.sleep(for: delayBetweenAttempts)
                }
            }
        }
        throw lastError!
    }
}
