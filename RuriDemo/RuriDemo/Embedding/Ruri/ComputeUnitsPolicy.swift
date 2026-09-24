import CoreML

/// Which compute units a CoreML model is allowed to run on.
enum ComputeUnitsPolicy {
    /// Every accelerator on a real device (CPU/GPU/Neural Engine). The iOS
    /// Simulator is pinned to CPU-only: on some macOS betas its MPSGraph
    /// backend fails for this model and falls back silently instead of
    /// erroring.
    case `default`

    func resolved() -> MLComputeUnits {
        #if targetEnvironment(simulator)
        return .cpuOnly
        #else
        return .all
        #endif
    }
}
