import NaturalLanguage

/// Sample data for the model-comparison demo: a query about feeling nostalgic
/// while looking at the moon, and a set of documents designed to probe
/// whether a model is matching on *meaning* rather than the literal
/// character "月" (moon) — spanning modern Japanese, classical Japanese,
/// English, near-miss distractors, and control documents that share no
/// topic with the query at all.
enum MoonNostalgiaDataset {
    static let queryJapanese = "月を見て懐かしい気持ちになる話"
    static let queryEnglish = "A story about feeling nostalgic while looking at the moon"

    static let documents: [Document] = [
        Document(
            label: "現代文・直接一致",
            text: "夜空に浮かぶ月を眺めていたら、昔のことが思い出されて胸が熱くなった。",
            language: .japanese
        ),
        Document(
            label: "現代文・月見の思い出",
            text: "満月を見ていると、子供の頃に家族で月見をした夜を思い出す。",
            language: .japanese
        ),
        Document(
            label: "現代文・望郷",
            text: "窓から見える月がとても綺麗で、なぜか故郷を思い出して懐かしくなった。",
            language: .japanese
        ),
        Document(
            label: "古典・芭蕉(月の句)",
            text: "名月や池をめぐりて夜もすがら",
            language: .japanese
        ),
        Document(
            label: "古典・百人一首(紫式部)",
            text: "めぐり逢ひて見しやそれとも分かぬ間に雲隠れにし夜半の月かな",
            language: .japanese
        ),
        Document(
            label: "古典・徒然草",
            text: "花はさかりに、月はくまなきをのみ見るものかは。",
            language: .japanese
        ),
        Document(
            label: "古典・竹取物語(かぐや姫)",
            text: "かぐや姫、月の顔見るは、忌むこととて制しけれども、ともすれば人間にも月をあはれがりたまふ。",
            language: .japanese
        ),
        Document(
            label: "祇園精舎(無関係・古語)",
            text: "祇園精舎の鐘の声、諸行無常の響きあり。娑羅双樹の花の色、盛者必衰の理をあらはす。",
            language: .japanese
        ),
        Document(
            label: "味噌ラーメン(無関係)",
            text: "この味噌ラーメンは煮干しだしが効いていて、とても美味しかった。",
            language: .japanese
        ),
        Document(
            label: "今日の天気(無関係)",
            text: "今日の東京の天気は晴れ、最高気温は26度の見込みです。",
            language: .japanese
        ),
        Document(
            label: "現代・間接(満ち欠け、月の字なし)",
            text: "満ち欠けを繰り返しながら夜空を渡っていくあの白い光を、幼い頃はよく縁側から眺めたものだ。",
            language: .japanese
        ),
        Document(
            label: "現代・間接(十五夜、月の字なし)",
            text: "十五夜の夜、すすきを飾って夜空に浮かぶ丸い光を眺めるのが、祖母の家での毎年の習わしだった。",
            language: .japanese
        ),
        Document(
            label: "古典風・間接(望の夜、月の字なし)",
            text: "望の夜、庭に出でて空を仰げば、昔のことのみ思ひ出でられける。",
            language: .japanese
        ),
        Document(
            label: "古典風・間接(玉兎、月の字なし)",
            text: "玉兎西へ傾く頃、庭に佇みて往時を偲びけり。",
            language: .japanese
        ),
        Document(
            label: "近い誤答・太陽(懐かしさなし)",
            text: "朝、太陽が昇るのを見て、今日も一日頑張ろうと思った。",
            language: .japanese
        ),
        Document(
            label: "英語・直接一致",
            text: "Looking at the full moon tonight brought back memories of my childhood home.",
            language: .english
        ),
        Document(
            label: "英語・望郷",
            text: "Every time I see the moon rise, I feel a quiet longing for the town where I grew up.",
            language: .english
        ),
        Document(
            label: "英語・詩的表現",
            text: "The old moon always makes me think of summers long gone.",
            language: .english
        ),
        Document(
            label: "英語・無関係",
            text: "The quarterly earnings report showed a 12% increase in revenue.",
            language: .english
        ),
    ]

    static func documents(in language: NLLanguage) -> [Document] {
        documents.filter { $0.language == language }
    }
}
