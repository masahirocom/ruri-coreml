import CoreML

/// Builds the `MLFeatureProvider` ruri-v3's CoreML graph expects
/// (`input_ids` / `attention_mask`, both `[1, sequenceLength]` int32).
enum RuriModelInputBuilder {
    /// CoreML input tensors carry an explicit batch dimension; this app only
    /// ever embeds one string at a time.
    private static let batchSize = 1

    private static let inputIDsFeatureName = "input_ids"
    private static let attentionMaskFeatureName = "attention_mask"

    static func makeFeatureProvider(
        for encoding: RuriTokenEncoding,
        sequenceLength: Int
    ) throws -> MLFeatureProvider {
        let shape: [NSNumber] = [NSNumber(value: batchSize), NSNumber(value: sequenceLength)]

        let inputIDs = try MLMultiArray(shape: shape, dataType: .int32)
        let attentionMask = try MLMultiArray(shape: shape, dataType: .int32)
        for index in 0..<sequenceLength {
            inputIDs[index] = NSNumber(value: encoding.paddedTokenIDs[index])
            attentionMask[index] = NSNumber(value: encoding.attentionMask[index])
        }

        return try MLDictionaryFeatureProvider(dictionary: [
            inputIDsFeatureName: MLFeatureValue(multiArray: inputIDs),
            attentionMaskFeatureName: MLFeatureValue(multiArray: attentionMask),
        ])
    }
}
