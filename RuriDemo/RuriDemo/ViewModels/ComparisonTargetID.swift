/// Identifies one column of the model comparison (which embedder, shown
/// under which tab). `CaseIterable` is what lets the view model and views
/// loop over "every comparison target" instead of hand-listing three of them.
enum ComparisonTargetID: CaseIterable, Identifiable {
    case ruri
    case appleJapanese
    case appleEnglish

    var id: Self { self }

    var displayName: String {
        switch self {
        case .ruri: "ruri-v3"
        case .appleJapanese: "Apple(日本語)"
        case .appleEnglish: "Apple(英語)"
        }
    }

    var loadingStatusText: String {
        switch self {
        case .ruri:
            "ruri-v3 (CoreML) を読み込み中…"
        case .appleJapanese:
            "Apple NLContextualEmbedding (日本語) を読み込み中…(初回はモデルのダウンロードが走ります)"
        case .appleEnglish:
            "Apple NLContextualEmbedding (英語) を読み込み中…"
        }
    }
}
