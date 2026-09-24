import SwiftUI

/// Names exactly which model is backing the currently-shown results, so
/// there's never ambiguity about which build/configuration produced them
/// (this mattered in practice while swapping model variants during testing).
struct ActiveModelBadge: View {
    let modelDescription: String?

    var body: some View {
        if let modelDescription {
            Text("使用モデル: \(modelDescription)")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }
}
