import NaturalLanguage

/// A single piece of sample text used as a similarity-search candidate,
/// tagged with the language it should be compared in (Apple's per-language
/// embedding models cannot mix languages within one comparison).
struct Document: Identifiable {
    let id = UUID()
    let label: String
    let text: String
    let language: NLLanguage
}
