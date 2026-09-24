import Foundation

/// Drives the model-comparison screen: loads every `ComparisonTargetID`'s
/// embedder, then runs the same similarity search across all of them when
/// asked. Holds no model- or search-specific logic itself — that lives in
/// `ComparisonTargetLoader` and `SimilaritySearchService` respectively — so
/// this class's only job is coordinating state for the view.
@MainActor
final class ComparisonSearchViewModel: ObservableObject {
    @Published private(set) var resultsByTarget: [ComparisonTargetID: [SimilarityResult]] = [:]
    @Published private(set) var isLoadingModels = false
    @Published private(set) var isSearching = false
    @Published private(set) var loadingStatusText = ""
    @Published private(set) var errorMessages: [String] = []

    private var loadedTargets: [ComparisonTargetID: LoadedComparisonTarget] = [:]
    private let searchService = SimilaritySearchService()

    func results(for target: ComparisonTargetID) -> [SimilarityResult] {
        resultsByTarget[target] ?? []
    }

    func loadModels() async {
        isLoadingModels = true
        errorMessages = []

        for id in ComparisonTargetID.allCases {
            loadingStatusText = id.loadingStatusText
            do {
                loadedTargets[id] = try await ComparisonTargetLoader.load(id)
            } catch {
                debugLog("[ComparisonSearchViewModel] \(id) failed to load: \(error)")
                errorMessages.append("\(id.displayName)読み込みエラー: \(error.localizedDescription)")
            }
        }

        loadingStatusText = ""
        isLoadingModels = false
    }

    func search() async {
        isSearching = true
        errorMessages = []

        for id in ComparisonTargetID.allCases {
            guard let target = loadedTargets[id] else { continue }
            do {
                resultsByTarget[id] = try await searchService.rank(
                    query: target.query,
                    against: target.documents,
                    queryEmbedder: target.queryEmbedder,
                    documentEmbedder: target.documentEmbedder
                )
            } catch {
                debugLog("[ComparisonSearchViewModel] \(id) search failed: \(error)")
                errorMessages.append("\(id.displayName)推論エラー: \(error.localizedDescription)")
            }
        }

        isSearching = false
    }
}
