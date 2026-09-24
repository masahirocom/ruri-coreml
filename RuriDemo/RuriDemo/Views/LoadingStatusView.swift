import SwiftUI

/// Shown in place of the search UI while every comparison target's model
/// is still loading.
struct LoadingStatusView: View {
    let statusText: String

    private static let fallbackText = "読み込み中…"

    var body: some View {
        ProgressView(statusText.isEmpty ? Self.fallbackText : statusText)
            .padding()
            .multilineTextAlignment(.center)
    }
}
