import Foundation

/// A document scored against a query by one embedding model.
struct SimilarityResult: Identifiable {
    let id = UUID()
    let document: Document
    let score: Float
}
