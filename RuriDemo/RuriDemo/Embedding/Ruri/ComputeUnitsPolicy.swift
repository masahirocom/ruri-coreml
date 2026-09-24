import CoreML

/// Which compute units a CoreML model is allowed to run on.
enum ComputeUnitsPolicy {
    /// Uses every accelerator normally available on the running platform,
    /// except that the iOS Simulator is pinned to CPU-only.
    ///
    /// Background: on at least one macOS 27 beta, the Simulator's CoreML
    /// backend fails to validate its MPSGraph (GPU/ANE) path for this model
    /// and falls back silently rather than erroring, which is worse than
    /// just avoiding it. Real devices are unaffected and get full
    /// acceleration. If a future OS fixes this, drop the `#if` branch below.
    case `default`

    /// Forces CPU execution regardless of platform. Useful for debugging
    /// numerical differences between compute backends.
    case cpuOnly

    /// Uses every accelerator normally available, unconditionally. Only
    /// safe once the Simulator issue described above is confirmed fixed.
    case allAvailable

    func resolved() -> MLComputeUnits {
        switch self {
        case .cpuOnly:
            return .cpuOnly
        case .allAvailable:
            return .all
        case .default:
            #if targetEnvironment(simulator)
            return .cpuOnly
            #else
            return .all
            #endif
        }
    }
}
