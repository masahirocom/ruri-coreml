import CoreML

extension MLMultiArray {
    /// Copies this array's contents out as `[Float]`.
    ///
    /// `NSNumber`'s `.floatValue` on a `.float16`-typed array has been
    /// observed to return NaN on real devices while working by coincidence
    /// in the Simulator (where CoreML's CPU backend appears to compute in
    /// float32 regardless of the declared storage type). Float16 arrays are
    /// therefore read directly through their native `Float16` byte layout;
    /// every other dtype goes through the standard `NSNumber` path.
    func asFloatVector() -> [Float] {
        guard dataType == .float16 else {
            return (0..<count).map { self[$0].floatValue }
        }
        let float16Pointer = dataPointer.bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(float16Pointer[$0]) }
    }
}
