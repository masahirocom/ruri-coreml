import Foundation

/// Runs `operation` and reports how long it took alongside its result, so
/// callers that want to log latency don't each reimplement a `DispatchTime`
/// pair.
func measuringElapsedMilliseconds<T>(_ operation: () throws -> T) rethrows -> (result: T, milliseconds: Double) {
    let start = DispatchTime.now()
    let result = try operation()
    let elapsedNanoseconds = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    let nanosecondsPerMillisecond = 1_000_000.0
    return (result, Double(elapsedNanoseconds) / nanosecondsPerMillisecond)
}
