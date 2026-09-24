import Foundation

/// Logs a message only in debug builds. Release builds compile this to a no-op,
/// so nothing here ever reaches a shipped binary's logs.
func debugLog(_ message: @autoclosure () -> String) {
    #if DEBUG
    NSLog("%@", message())
    #endif
}
