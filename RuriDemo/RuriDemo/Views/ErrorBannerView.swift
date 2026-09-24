import SwiftUI

/// Renders zero or more error strings as a small red block. Renders nothing
/// when `messages` is empty, so callers don't need to branch on that.
struct ErrorBannerView: View {
    let messages: [String]

    var body: some View {
        if !messages.isEmpty {
            VStack(alignment: .leading, spacing: 2) {
                ForEach(messages, id: \.self) { message in
                    Text(message)
                        .foregroundColor(.red)
                        .font(.footnote)
                }
            }
            .padding(.horizontal)
        }
    }
}
