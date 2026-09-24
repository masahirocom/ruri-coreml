/// A comparison target's runtime state once its embedder has finished
/// loading: the query/document set it searches over, and the embedder(s)
/// used for each side of the search.
struct LoadedComparisonTarget {
    let id: ComparisonTargetID
    let query: String
    let documents: [Document]
    let queryEmbedder: any SentenceEmbedder
    let documentEmbedder: any SentenceEmbedder

    /// What's actually loaded and running — shown in the UI so it's never
    /// ambiguous which model backs a given tab's results (e.g. after
    /// swapping `RuriModelConfiguration` during testing).
    let modelDescription: String
}
