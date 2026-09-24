import SwiftUI

/// One named section of ranked results, or a placeholder if there are none yet.
struct SimilarityResultListView: View {
    let title: String
    let results: [SimilarityResult]

    var body: some View {
        Section(title) {
            if results.isEmpty {
                Text("(結果なし)")
                    .foregroundColor(.secondary)
                    .font(.footnote)
            }
            ForEach(results) { result in
                SimilarityResultRow(result: result)
            }
        }
    }
}
