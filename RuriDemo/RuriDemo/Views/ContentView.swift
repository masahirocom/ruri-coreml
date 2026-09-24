import SwiftUI

/// Top-level layout for the model-comparison screen. Owns no business logic
/// of its own — everything here is either state from `ComparisonSearchViewModel`
/// or delegated to a dedicated subview.
struct ContentView: View {
    @StateObject private var viewModel = ComparisonSearchViewModel()
    @State private var selectedTarget: ComparisonTargetID = .ruri

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                if viewModel.isLoadingModels {
                    LoadingStatusView(statusText: viewModel.loadingStatusText)
                } else {
                    QueryHeaderView(
                        queryJapanese: MoonNostalgiaDataset.queryJapanese,
                        queryEnglish: MoonNostalgiaDataset.queryEnglish
                    )

                    SearchActionButton(isSearching: viewModel.isSearching) {
                        Task { await viewModel.search() }
                    }

                    ErrorBannerView(messages: viewModel.errorMessages)

                    ComparisonTabPicker(selection: $selectedTarget)

                    if selectedTarget == .ruri {
                        RuriModelPicker(selection: viewModel.selectedRuriConfiguration) { configuration in
                            Task { await viewModel.selectRuriModel(configuration) }
                        }
                    }
                    ActiveModelBadge(modelDescription: viewModel.modelDescription(for: selectedTarget))

                    List {
                        SimilarityResultListView(
                            title: selectedTarget.displayName,
                            results: viewModel.results(for: selectedTarget)
                        )
                    }
                    .listStyle(.insetGrouped)
                }
            }
            .navigationTitle("ruri-v3 vs Apple")
            .task {
                await viewModel.loadModels()
            }
        }
    }
}

#Preview {
    ContentView()
}
