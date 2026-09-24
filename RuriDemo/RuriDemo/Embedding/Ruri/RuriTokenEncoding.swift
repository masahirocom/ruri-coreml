/// Token ids and attention mask, both padded/truncated to a fixed length —
/// exactly the shape ruri-v3's CoreML graph expects as input.
struct RuriTokenEncoding {
    let paddedTokenIDs: [Int]
    let attentionMask: [Int]

    private static let paddingTokenID = 0
    private static let realTokenMaskValue = 1
    private static let paddingTokenMaskValue = 0

    /// Truncates `tokenIDs` to `sequenceLength` if needed, or pads it out
    /// with `paddingTokenID`, producing a matching attention mask.
    init(tokenIDs: [Int], sequenceLength: Int) {
        let truncated = tokenIDs.count > sequenceLength ? Array(tokenIDs.prefix(sequenceLength)) : tokenIDs
        let paddingCount = sequenceLength - truncated.count

        paddedTokenIDs = truncated + Array(repeating: Self.paddingTokenID, count: paddingCount)
        attentionMask =
            Array(repeating: Self.realTokenMaskValue, count: truncated.count)
            + Array(repeating: Self.paddingTokenMaskValue, count: paddingCount)
    }
}
