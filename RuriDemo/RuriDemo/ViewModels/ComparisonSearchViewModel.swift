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
    @Published private(set) var selectedRuriConfiguration: RuriModelConfiguration = .default

    private var loadedTargets: [ComparisonTargetID: LoadedComparisonTarget] = [:]
    private let searchService = SimilaritySearchService()

    func results(for target: ComparisonTargetID) -> [SimilarityResult] {
        resultsByTarget[target] ?? []
    }

    /// What's actually loaded for this tab (e.g. "ruri-v3-130m_seq128_int8"),
    /// so it's never ambiguous which model produced the results on screen.
    func modelDescription(for target: ComparisonTargetID) -> String? {
        loadedTargets[target]?.modelDescription
    }

    func loadModels() async {
        isLoadingModels = true
        errorMessages = []

        for id in ComparisonTargetID.allCases {
            await loadTarget(id)
        }

        loadingStatusText = ""
        isLoadingModels = false
    }

    /// Reloads just the ruri-v3 tab with a different bundled model variant
    /// (e.g. chosen from `RuriModelPicker`), leaving the Apple tabs alone.
    func selectRuriModel(_ configuration: RuriModelConfiguration) async {
        guard configuration != selectedRuriConfiguration else { return }
        selectedRuriConfiguration = configuration
        resultsByTarget[.ruri] = nil

        isLoadingModels = true
        await loadTarget(.ruri)
        loadingStatusText = ""
        isLoadingModels = false
    }

    private func loadTarget(_ id: ComparisonTargetID) async {
        loadingStatusText = id.loadingStatusText
        errorMessages.removeAll { $0.hasPrefix(id.displayName) }
        do {
            loadedTargets[id] = try await ComparisonTargetLoader.load(id, ruriConfiguration: selectedRuriConfiguration)
        } catch {
            debugLog("[ComparisonSearchViewModel] \(id) failed to load: \(error)")
            errorMessages.append("\(id.displayName)読み込みエラー: \(error.localizedDescription)")
        }
    }

    func search() async {
        isSearching = true
        errorMessages = []

        for id in ComparisonTargetID.allCases {
            guard let target = loadedTargets[id] else { continue }
            do {
                let results = try await searchService.rank(
                    query: target.query,
                    against: target.documents,
                    queryEmbedder: target.queryEmbedder,
                    documentEmbedder: target.documentEmbedder
                )
                resultsByTarget[id] = results
                logResults(results, for: target)
            } catch {
                debugLog("[ComparisonSearchViewModel] \(id) search failed: \(error)")
                errorMessages.append("\(id.displayName)推論エラー: \(error.localizedDescription)")
            }
        }

        isSearching = false
    }

    private func logResults(_ results: [SimilarityResult], for target: LoadedComparisonTarget) {
        debugLog("[Results] \(target.modelDescription) query=\(target.query)")
        for (index, result) in results.enumerated() {
            let rank = index + 1
            debugLog("[Results] \(rank)\t\(String(format: "%.4f", result.score))\t\(result.document.label)")
        }
    }
}
