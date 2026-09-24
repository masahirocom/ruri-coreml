import Foundation

/// Pure vector-math helpers shared by every embedder. Kept free of any model-
/// or framework-specific knowledge so they can be tested and reasoned about
/// on their own.
enum VectorMath {
    /// The dot product of two equal-length unit vectors is their cosine similarity.
    static func dotProduct(_ lhs: [Float], _ rhs: [Float]) -> Float {
        precondition(lhs.count == rhs.count, "Vectors must have the same dimensionality to compare.")
        var sum: Float = 0
        for index in lhs.indices { sum += lhs[index] * rhs[index] }
        return sum
    }

    /// Element-wise mean of a non-empty list of equal-length vectors.
    static func mean(of vectors: [[Double]]) -> [Double] {
        precondition(!vectors.isEmpty, "Cannot average an empty list of vectors.")
        let dimension = vectors[0].count
        var sum = [Double](repeating: 0, count: dimension)
        for vector in vectors {
            for index in 0..<dimension { sum[index] += vector[index] }
        }
        let count = Double(vectors.count)
        return sum.map { $0 / count }
    }

    /// Scales `vector` to unit length. Leaves an all-zero vector unchanged
    /// rather than dividing by zero.
    static func l2Normalized(_ vector: [Float]) -> [Float] {
        let magnitude = sqrt(vector.reduce(Float(0)) { $0 + $1 * $1 })
        guard magnitude > .leastNormalMagnitude else { return vector }
        return vector.map { $0 / magnitude }
    }
}
