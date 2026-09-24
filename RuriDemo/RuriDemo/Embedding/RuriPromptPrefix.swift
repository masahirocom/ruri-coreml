/// ruri-v3 was trained with a "1+3 prefix scheme": the same encoder produces
/// better embeddings when the input text is prefixed to say what role it
/// plays. See https://huggingface.co/cl-nagoya/ruri-v3-130m for the scheme
/// this mirrors.
enum RuriPromptPrefix: String {
    /// General-purpose semantic similarity, no declared role.
    case none = ""
    /// Text used for classification / clustering.
    case topic = "トピック: "
    /// The query side of a search.
    case searchQuery = "検索クエリ: "
    /// The document side of a search.
    case searchDocument = "検索文書: "

    func applied(to text: String) -> String {
        rawValue + text
    }
}
