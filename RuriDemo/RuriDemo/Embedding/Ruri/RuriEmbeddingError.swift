import Foundation

enum RuriEmbeddingError: Error, LocalizedError {
    case modelResourceNotFound(String)
    case tokenizerResourceNotFound(String)
    case missingOutputFeature(String)

    var errorDescription: String? {
        switch self {
        case let .modelResourceNotFound(name):
            "Bundled CoreML model '\(name).mlmodelc' was not found. Check it was added to the Xcode target."
        case let .tokenizerResourceNotFound(name):
            "Bundled tokenizer folder '\(name)' was not found. Check it was added as a folder reference."
        case let .missingOutputFeature(name):
            "The CoreML model did not return an output named '\(name)'."
        }
    }
}
