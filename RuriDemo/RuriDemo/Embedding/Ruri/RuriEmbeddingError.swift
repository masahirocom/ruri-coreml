import Foundation

enum RuriEmbeddingError: Error, LocalizedError {
    case modelResourceNotFound(String)
    case tokenizerResourceNotFound(String)
    case missingOutputFeature(String)
    case nanOutput(modelResourceName: String, nanCount: Int, dimensionCount: Int)

    var errorDescription: String? {
        switch self {
        case let .modelResourceNotFound(name):
            "Bundled CoreML model '\(name).mlmodelc' was not found. Check it was added to the Xcode target."
        case let .tokenizerResourceNotFound(name):
            "Bundled tokenizer folder '\(name)' was not found. Check it was added as a folder reference."
        case let .missingOutputFeature(name):
            "The CoreML model did not return an output named '\(name)'."
        case let .nanOutput(modelResourceName, nanCount, dimensionCount):
            "\(modelResourceName) returned \(nanCount)/\(dimensionCount) NaN values. " +
            "This is a known issue with float16-compute and int8-quantized ruri-v3 " +
            "Core ML exports on real devices — see REPORT.md. Use the fp32 (unquantized) " +
            "model on-device."
        }
    }
}
