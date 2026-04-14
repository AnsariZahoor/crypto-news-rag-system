"""
Hand-curated evaluation dataset for the crypto news RAG pipeline.

Each entry has:
    question      → what a user would ask the chatbot
    ground_truth  → the correct answer (used by RAGAS for recall/precision scoring)
    category      → topic bucket (for grouped metric analysis)
    difficulty    → easy / medium / hard (for stratified evaluation)

"""

from typing import Literal


Category   = Literal[
    "bitcoin", "ethereum", "defi", "regulation",
    "security", "stablecoins", "market", "institutional", "other"
]
Difficulty = Literal["easy", "medium", "hard"]


EVAL_DATASET: list[dict] = [

    # ── Bitcoin ─────────────────────────────────────────────────────────

    {
        "question": "What is the current state of Bitcoin spot demand?",
        "ground_truth": (
            "Bitcoin spot demand is in deep contraction according to onchain analytics "
            "firm CryptoQuant, even as some institutional buying continues."
        ),
        "category": "bitcoin",
        "difficulty": "easy",
    },
    {
        "question": "What does K33 say about Bitcoin's prolonged consolidation?",
        "ground_truth": (
            "K33 research and brokerage firm says Bitcoin's prolonged consolidation could "
            "signal a shift in market structure, with easing selling pressure pointing to "
            "a potential bottom."
        ),
        "category": "bitcoin",
        "difficulty": "medium",
    },
    {
        "question": "What happened during the Bitcoin two-block reorganization at heights 941881 and 941882?",
        "ground_truth": (
            "Bitcoin experienced a rare two-block reorganization at heights 941881 and 941882 "
            "after rival mining pools briefly produced parallel versions of the blockchain."
        ),
        "category": "bitcoin",
        "difficulty": "hard",
    },
    {
        "question": "What is CryptoQuant's estimated Bitcoin bear market bottom?",
        "ground_truth": (
            "CryptoQuant says Bitcoin's ultimate bear market bottom is currently around $55,000, "
            "and that bear market bottoms typically take months to form rather than occur instantly."
        ),
        "category": "bitcoin",
        "difficulty": "medium",
    },
    {
        "question": "What did MARA Holdings change about its treasury policy for 2026?",
        "ground_truth": (
            "MARA Holdings, the largest public bitcoin miner by BTC held, widened its treasury "
            "policy for 2026 to include the potential sale of its accumulated bitcoin reserves, "
            "according to a 10-K filing."
        ),
        "category": "bitcoin",
        "difficulty": "medium",
    },

    # ── Ethereum ─────────────────────────────────────────────────────────

    {
        "question": "How much ETH has the Ethereum Foundation deployed into staking?",
        "ground_truth": (
            "The Ethereum Foundation doubled its staked ETH to 47,050 ETH, worth roughly "
            "$96.6 million, pushing it past the two-thirds mark of its 70,000 ETH goal."
        ),
        "category": "ethereum",
        "difficulty": "easy",
    },

    # ── DeFi ─────────────────────────────────────────────────────────────

    {
        "question": "Why is best execution no longer guaranteed by sticking to a single venue in DeFi?",
        "ground_truth": (
            "Liquidity that once lived in a handful of places now spans thousands of pools "
            "and a growing set of DEX venues across multiple networks, each offering different "
            "prices and liquidity depth, making single-venue execution suboptimal."
        ),
        "category": "defi",
        "difficulty": "medium",
    },
    {
        "question": "What is intent-centric design in crypto protocols?",
        "ground_truth": (
            "Intent-centric design means users submit what outcome they want rather than "
            "the full methodology of how an operation should be executed, instead of directly "
            "submitting transactions with explicit execution steps."
        ),
        "category": "defi",
        "difficulty": "medium",
    },
    {
        "question": "What is mEVUSD and who developed it?",
        "ground_truth": (
            "mEVUSD is a regulatory-compliant USDC-denominated tokenized strategy co-developed "
            "by Apollo Crypto with noncustodial staking infrastructure provider Everstake and "
            "an onchain investment manager."
        ),
        "category": "defi",
        "difficulty": "hard",
    },

    # ── Stablecoins ───────────────────────────────────────────────────────

    {
        "question": "How are stablecoins transforming the global financial system?",
        "ground_truth": (
            "Stablecoins, once limited in use, are now a key part of a fundamental transformation "
            "of the interconnected global financial system, with use cases poised to expand "
            "the market significantly according to The Block Research."
        ),
        "category": "stablecoins",
        "difficulty": "easy",
    },

    # ── Regulation ────────────────────────────────────────────────────────

    {
        "question": "What did Alabama's governor approve regarding DAOs?",
        "ground_truth": (
            "Alabama Governor Kay Ivey gave final approval to a bill that legally recognizes "
            "decentralized autonomous organization-like structures in the state."
        ),
        "category": "regulation",
        "difficulty": "easy",
    },
    {
        "question": "What did SEC Chairman Paul Atkins say about crypto oversight?",
        "ground_truth": (
            "SEC Chairman Paul Atkins described the prior administration's approach to crypto "
            "as a big missed opportunity and said the SEC is working to regain momentum "
            "on crypto oversight."
        ),
        "category": "regulation",
        "difficulty": "medium",
    },
    {
        "question": "What is South Korea's proposed law about crypto finfluencers?",
        "ground_truth": (
            "South Korea's ruling political party introduced a bill requiring social media "
            "finfluencers to disclose their personal asset holdings and any compensation "
            "received when promoting cryptocurrencies."
        ),
        "category": "regulation",
        "difficulty": "medium",
    },
    {
        "question": "What inquiry was opened into Binance by the US Senate?",
        "ground_truth": (
            "Senator Richard Blumenthal, the top Democrat on an investigative panel within "
            "the Senate Homeland Security Committee, opened an inquiry into crypto giant Binance "
            "following reports of potential sanctions violations."
        ),
        "category": "regulation",
        "difficulty": "hard",
    },
    {
        "question": "What charter did Coinbase receive from the OCC?",
        "ground_truth": (
            "Coinbase received conditional approval for a national trust company charter from "
            "the Office of the Comptroller of the Currency, aimed at bringing federal regulatory "
            "uniformity to custody and market infrastructure."
        ),
        "category": "regulation",
        "difficulty": "medium",
    },

    # ── Security ──────────────────────────────────────────────────────────

    {
        "question": "How many security incidents occurred on the Web2 side in 2024?",
        "ground_truth": (
            "In 2024 there were more than 30,000 security incidents and 10,000 data breaches "
            "on the Web2 side, highlighting ongoing vulnerabilities across both Web2 and Web3."
        ),
        "category": "security",
        "difficulty": "easy",
    },
    {
        "question": "What is approval-phishing and what operation targets it?",
        "ground_truth": (
            "Approval-phishing is a type of fraud that drains crypto wallets by tricking victims "
            "into approving malicious transactions. Law enforcement agencies from the U.S., UK, "
            "and Canada are targeting it through Operation Atlantic."
        ),
        "category": "security",
        "difficulty": "medium",
    },
    {
        "question": "What does Google's latest quantum computing research mean for Bitcoin?",
        "ground_truth": (
            "Google's latest quantum computing research is reigniting debate about whether quantum "
            "computing could one day threaten Bitcoin and other blockchain networks, and how soon "
            "developers need to act to prepare defenses."
        ),
        "category": "security",
        "difficulty": "hard",
    },

    # ── Institutional ─────────────────────────────────────────────────────

    {
        "question": "What does the Coinbase and Block Research Fortune 500 report say about blockchain adoption?",
        "ground_truth": (
            "Coinbase's Q2 2024 State of Crypto report, produced in collaboration with The Block, "
            "provides an updated view of blockchain adoption among America's top public companies, "
            "showing that Fortune 500 companies are moving onchain."
        ),
        "category": "institutional",
        "difficulty": "medium",
    },
    {
        "question": "Is Stripe interested in acquiring PayPal?",
        "ground_truth": (
            "According to Bloomberg, Stripe has expressed preliminary interest in acquiring all "
            "or part of PayPal. Stripe is a private payment processor that has increasingly "
            "leaned into crypto."
        ),
        "category": "institutional",
        "difficulty": "easy",
    },
    {
        "question": "What was Nvidia accused of concealing according to the class action lawsuit?",
        "ground_truth": (
            "A federal judge certified a class of investors accusing Nvidia and its CEO Jensen Huang "
            "of concealing the extent to which the company's gaming GPU revenue was driven by "
            "cryptocurrency mining demand."
        ),
        "category":   "institutional",
        "difficulty": "hard",
    },

    # ── Market ────────────────────────────────────────────────────────────

    {
        "question": "Why did Bitcoin price fall after its brief rally last week?",
        "ground_truth": (
            "The price of Bitcoin fell following a brief rally as a spike in oil prices "
            "weighed on Asian equity markets."
        ),
        "category": "market",
        "difficulty": "easy",
    },
    {
        "question": "What are the two fundamental structural barriers to blockchain adoption?",
        "ground_truth": (
            "The two barriers are: first, ideological tribalism that has fragmented the ecosystem "
            "by splitting developer talent, liquidity, and community focus across competing "
            "winner-take-all networks; and second, scaling limitations on public blockchains."
        ),
        "category": "market",
        "difficulty": "medium",
    },

    # ── Memecoin / Other ─────────────────────────────────────────────────

    {
        "question": "What happened to the Solana memecoin based on Jonathan the tortoise?",
        "ground_truth": (
            "A Solana-based memecoin launched in honor of Jonathan, a 193-year-old tortoise "
            "believed to be the oldest living land animal, rallied after news spread of "
            "his alleged passing."
        ),
        "category": "other",
        "difficulty": "easy",
    },
    {
        "question": "What does The Block Research report say about the future of consumer crypto?",
        "ground_truth": (
            "The consumer crypto landscape is poised to reshape how individuals interact with "
            "products and services in everyday life, with the report focusing on defining, mapping, "
            "and analyzing the emerging sector."
        ),
        "category": "other",
        "difficulty": "medium",
    },
]



def by_category(category: str) -> list[dict]:
    """Return all eval items for a specific category."""
    return [item for item in EVAL_DATASET if item["category"] == category]


def by_difficulty(difficulty: str) -> list[dict]:
    """Return all eval items for a specific difficulty level."""
    return [item for item in EVAL_DATASET if item["difficulty"] == difficulty]


def summary() -> None:
    """Print dataset statistics."""
    from collections import Counter

    cats = Counter(item["category"] for item in EVAL_DATASET)
    diffs = Counter(item["difficulty"] for item in EVAL_DATASET)

    print(f"Total questions : {len(EVAL_DATASET)}")
    print()
    print("By category:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat:<16} {count}")
    print()
    print("By difficulty:")
    for diff, count in sorted(diffs.items()):
        print(f"  {diff:<16} {count}")


if __name__ == "__main__":
    summary()