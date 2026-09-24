import SwiftUI

/// One ranked (score, document) pair.
struct SimilarityResultRow: View {
    let result: SimilarityResult

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(String(format: "%.4f", result.score))
                    .font(.caption)
                    .bold()
                    .foregroundColor(.blue)
                Text(result.document.label)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Text(result.document.text)
                .font(.footnote)
        }
        .padding(.vertical, 2)
    }
}
