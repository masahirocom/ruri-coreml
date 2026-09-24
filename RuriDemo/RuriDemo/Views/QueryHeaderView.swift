import SwiftUI

/// Displays the fixed Japanese/English demo queries the comparison runs.
struct QueryHeaderView: View {
    let queryJapanese: String
    let queryEnglish: String

    var body: some View {
        VStack(spacing: 4) {
            Text("クエリ(JA): \(queryJapanese)")
                .font(.footnote)
            Text("Query (EN): \(queryEnglish)")
                .font(.footnote)
                .foregroundColor(.secondary)
        }
    }
}
