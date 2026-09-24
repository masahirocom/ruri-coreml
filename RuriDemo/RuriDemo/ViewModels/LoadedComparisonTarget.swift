/// A comparison target's runtime state once its embedder has finished
/// loading: the query/document set it searches over, and the embedder(s)
/// used for each side of the search.
struct LoadedComparisonTarget {
    let id: ComparisonTargetID
    let query: String
    let documents: [Document]
    let queryEmbedder: any SentenceEmbedder
    let documentEmbedder: any SentenceEmbedder
}
