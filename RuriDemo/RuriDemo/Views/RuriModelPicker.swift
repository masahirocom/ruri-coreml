import SwiftUI

/// Lets the user switch which bundled ruri-v3 `.mlpackage` variant is
/// loaded, without editing source code — see
/// `RuriModelConfiguration.allKnownVariants`. Only meaningful while the
/// ruri-v3 tab is showing.
struct RuriModelPicker: View {
    let selection: RuriModelConfiguration
    let onSelect: (RuriModelConfiguration) -> Void

    var body: some View {
        Menu {
            ForEach(RuriModelConfiguration.allKnownVariants) { configuration in
                Button {
                    onSelect(configuration)
                } label: {
                    if configuration.isKnownToFailOnRealDevice {
                        Label(configuration.displayName, systemImage: "exclamationmark.triangle")
                    } else {
                        Text(configuration.displayName)
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Text("ruriモデル: \(selection.displayName)")
                Image(systemName: "chevron.up.chevron.down")
            }
            .font(.caption)
        }
    }
}
