import SwiftUI

/// The button that kicks off `ComparisonSearchViewModel.search()`, showing a
/// spinner in place of its label while a search is in flight.
struct SearchActionButton: View {
    let isSearching: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            if isSearching {
                ProgressView()
            } else {
                Text("比較検索を実行")
            }
        }
        .buttonStyle(.borderedProminent)
        .disabled(isSearching)
    }
}
