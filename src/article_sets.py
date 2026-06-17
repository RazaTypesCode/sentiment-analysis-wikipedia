"""
Curated article collections for comparative analysis.

Each set is a list of Wikipedia article titles. Keep groups roughly the same
size so statistical tests are fair. Titles must match Wikipedia exactly.
"""

# ── Conflicts ──────────────────────────────────────────────────────────────────

CIVIL_WARS_GLOBAL_SOUTH = [
    "Nigerian Civil War",
    "Sudanese Civil War (1983–2005)",
    "Second Congo War",
    "Angolan Civil War",
    "Mozambican Civil War",
    "Sierra Leone Civil War",
    "Liberian Civil War (1989–1997)",
    "Sri Lankan Civil War",
]

CIVIL_WARS_GLOBAL_NORTH = [
    "English Civil War",
    "Finnish Civil War",
    "Russian Civil War",
    "Spanish Civil War",
    "American Civil War",
    "French Wars of Religion",
    "Wars of the Roses",
    "Greek Civil War",
]

# ── Independence movements ─────────────────────────────────────────────────────

INDEPENDENCE_AFRICA = [
    "Algerian War",
    "Mau Mau rebellion",
    "Mozambican War of Independence",
    "Angolan War of Independence",
    "Guinean War of Independence",
    "Kenyan independence",
    "Ghanaian independence",
]

INDEPENDENCE_EUROPE = [
    "Irish War of Independence",
    "Greek War of Independence",
    "Hungarian Revolution of 1848",
    "Norwegian independence from Sweden",
    "Czech independence",
    "Polish independence",
    "Finnish Declaration of Independence",
]

# ── Political leaders ──────────────────────────────────────────────────────────

LEADERS_AFRICA = [
    "Nelson Mandela",
    "Kwame Nkrumah",
    "Patrice Lumumba",
    "Thomas Sankara",
    "Julius Nyerere",
    "Samora Machel",
    "Jomo Kenyatta",
]

LEADERS_EUROPE = [
    "Winston Churchill",
    "Charles de Gaulle",
    "Otto von Bismarck",
    "Napoleon Bonaparte",
    "Margaret Thatcher",
    "Franklin D. Roosevelt",
    "Abraham Lincoln",
]

# ── Scientists ─────────────────────────────────────────────────────────────────

SCIENTISTS_WESTERN = [
    "Isaac Newton",
    "Albert Einstein",
    "Marie Curie",
    "Nikola Tesla",
    "Charles Darwin",
    "Richard Feynman",
    "Max Planck",
]

SCIENTISTS_NON_WESTERN = [
    "Srinivasa Ramanujan",
    "Ibn al-Haytham",
    "Al-Khwarizmi",
    "Abdus Salam",
    "C. V. Raman",
    "Jagadish Chandra Bose",
    "Satyendra Nath Bose",
]

# ── Historical empires ─────────────────────────────────────────────────────────

EMPIRES_EUROPEAN = [
    "British Empire",
    "French colonial empire",
    "Spanish Empire",
    "Portuguese Empire",
    "Dutch Empire",
    "Roman Empire",
]

EMPIRES_ASIAN = [
    "Mongol Empire",
    "Ottoman Empire",
    "Mughal Empire",
    "Qing dynasty",
    "Han dynasty",
    "Maurya Empire",
]

# ── Thematic (topic-level comparison) ─────────────────────────────────────────

TOPICS_TECHNOLOGY = [
    "Artificial intelligence",
    "Internet",
    "Nuclear power",
    "Genetic engineering",
    "Space exploration",
    "Blockchain",
]

TOPICS_ENVIRONMENT = [
    "Climate change",
    "Deforestation",
    "Ocean acidification",
    "Biodiversity loss",
    "Renewable energy",
    "Carbon capture",
]

# ── Convenience registry ───────────────────────────────────────────────────────

SETS: dict[str, list[str]] = {
    "Civil Wars — Global South": CIVIL_WARS_GLOBAL_SOUTH,
    "Civil Wars — Global North": CIVIL_WARS_GLOBAL_NORTH,
    "Independence — Africa": INDEPENDENCE_AFRICA,
    "Independence — Europe": INDEPENDENCE_EUROPE,
    "Leaders — Africa": LEADERS_AFRICA,
    "Leaders — Europe": LEADERS_EUROPE,
    "Scientists — Western": SCIENTISTS_WESTERN,
    "Scientists — Non-Western": SCIENTISTS_NON_WESTERN,
    "Empires — European": EMPIRES_EUROPEAN,
    "Empires — Asian": EMPIRES_ASIAN,
    "Technology Topics": TOPICS_TECHNOLOGY,
    "Environment Topics": TOPICS_ENVIRONMENT,
}
