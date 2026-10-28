"""Stage 3: evidence-based messaging recommendations.

No public dataset has posts + comments, so we do NOT claim to measure which post
wording works. Instead each recommendation is a hypothesis to test, linked to the
numbers found in the comments, using simple rules (a playbook per argument type).
"""
from pyspark.sql import DataFrame, SparkSession

# Which advice goes with which kind of sexism (matched by words in the category name)
PLAYBOOK = [
    ("threat", "Do not engage. Hide or report the comments, pin a short community-rules note, "
               "and share support or reporting links."),
    ("derog", "Lead with real stories and named achievements instead of abstract claims; "
              "set and enforce a no-insults rule so discussion stays on the facts."),
    ("animos", "Avoid 'men versus women' wording. Use 'everyone benefits' framing and show "
               "men as allies in the examples you choose."),
    ("prejudic", "Stereotype-based arguments: use a 'myth vs. evidence' format with one clear "
                 "statistic and a source, and invite questions rather than debate."),
]
DEFAULT_ADVICE = "Run a small A/B test of two wordings and compare the comment tone."
TOPICS = ["gender_equality", "womens_education", "employment",
          "independence", "harassment_violence", "traditional_roles"]


def _advice(category: str) -> str:
    c = (category or "").lower()
    return next((text for key, text in PLAYBOOK if key in c), DEFAULT_ADVICE)


def build_recommendations(spark: SparkSession, neg_rates: DataFrame, sex_types: DataFrame,
                          stance_split: DataFrame) -> DataFrame:
    """Combine the observed tables (small, so collected) into one row per topic."""
    neg = {(r["source"], r["topic"]): r for r in neg_rates.collect()}
    stance = {(r["topic"], r["label"]): r["pct_in_topic"] for r in stance_split.collect()}
    top_type = {}
    for r in sex_types.collect():                      # already ordered by n desc within topic
        top_type.setdefault(r["topic"], (r["sub_label"], r["pct_in_topic"], r["low_n"]))

    rows = []
    for t in TOPICS:
        sexist = neg.get(("edos", t))
        toxic = neg.get(("davidson", t))
        category, share, low = top_type.get(t, (None, None, True))
        evidence = (
            f"EDOS: {sexist['pct_negative'] if sexist else 'n/a'}% sexist "
            f"(avg {sexist['pct_negative_overall'] if sexist else 'n/a'}%); "
            f"Davidson: {toxic['pct_negative'] if toxic else 'n/a'}% hateful/offensive; "
            f"Stance: {stance.get((t, 'against'), 'n/a')}% against. "
            f"Most common sexism type: {category} ({share}% of sexist comments)."
        )
        rows.append((
            t,
            sexist["n"] if sexist else 0,
            sexist["lift"] if sexist else None,
            category,
            bool(low),
            _advice(category),
            evidence,
        ))
    cols = ["topic", "n_edos_comments", "sexism_lift", "dominant_sexism_type",
            "low_evidence", "recommendation", "evidence"]
    return spark.createDataFrame(rows, cols).orderBy("sexism_lift", ascending=False)