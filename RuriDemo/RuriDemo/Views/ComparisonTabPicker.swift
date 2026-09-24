import SwiftUI

/// Segmented control for switching which comparison target's results are
/// shown, so results can be compared one model at a time instead of in one
/// long scrolling list.
struct ComparisonTabPicker: View {
    @Binding var selection: ComparisonTargetID

    var body: some View {
        Picker("結果", selection: $selection) {
            ForEach(ComparisonTargetID.allCases) { target in
                Text(target.displayName).tag(target)
            }
        }
        .pickerStyle(.segmented)
        .padding(.horizontal)
    }
}
