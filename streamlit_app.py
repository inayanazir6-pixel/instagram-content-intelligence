"""Instagram Content Intelligence - web app (no Spark needed).

Reads the small result files made by the Spark pipeline:
  data/gold/*.csv         analysis tables
  models/exports/*.json   trained models, scored with plain Python
Run locally:  streamlit run streamlit_app.py
"""
import pandas as pd
import streamlit as st

from src import config
from src.scoring import analyze_comment

st.set_page_config(page_title="Instagram Content Intelligence", page_icon="📊", layout="wide")

STANCE_NAMES = {"favor": "Supportive", "none": "Neutral", "against": "Critical"}
TOPIC_NAMES = {
    "gender_equality": "Gender equality", "womens_education": "Women's education",
    "employment": "Employment", "independence": "Independence",
    "harassment_violence": "Harassment & violence", "traditional_roles": "Traditional gender roles",
    "untagged": "No topic matched",
}
TYPE_NAMES = {
    "derogation": "Derogation (insults)", "animosity": "Animosity (hostility)",
    "prejudiced_discussions": "Prejudiced discussion (stereotypes)",
    "threats_plans_to_harm_and_incitement": "Threats / incitement",
}


@st.cache_data
def gold(name: str) -> pd.DataFrame:
    df = pd.read_csv(config.GOLD_DIR / f"{name}.csv")
    for col in ("topic", "dominant_sexism_type"):
        if col in df:
            df[col] = df[col].map(lambda v: TOPIC_NAMES.get(v, v))
    return df


def pct(p: float) -> str:
    return f"{100 * p:.0f}%"


st.title("📊 Instagram Content Intelligence")
st.caption("Analysing how audiences react to gender and women's-rights content. "
           "Built with PySpark (training) and Streamlit (this app).")

tab_try, tab_topics, tab_reco, tab_models, tab_about = st.tabs(
    ["Try a comment", "Topic insights", "Recommendations", "Model quality", "About & limits"])

# ---- Try a comment -------------------------------------------------------------
with tab_try:
    text = st.text_area("Paste a comment (English):", height=110,
                        placeholder="e.g. Girls deserve the same education and pay as boys.")
    toxic_cut = st.slider("Mark as 'Toxic' when the toxicity model is at least this sure",
                          0.5, 0.95, 0.8, 0.05)
    if st.button("Analyse", type="primary") and text.strip():
        res = analyze_comment(text)
        if res["too_short"]:
            st.warning("Please write at least two words.")
        else:
            p_toxic = 1 - res["toxicity"]["probabilities"].get("neither", 0)
            stance = res["stance"]["label"]
            reaction = "Toxic" if p_toxic >= toxic_cut else STANCE_NAMES[stance]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall reaction", reaction)
            c2.metric("Stance", STANCE_NAMES[stance], pct(res["stance"]["probabilities"][stance]))
            c3.metric("Toxicity", res["toxicity"]["label"].replace("_", " "), pct(p_toxic) + " hateful/offensive")
            sx = res["sexism"]
            c4.metric("Sexist?", sx["label"], pct(sx["probabilities"][sx["label"]]))
            if "sexism_type" in res:
                st.info("Likely type of sexism: **" + TYPE_NAMES.get(res["sexism_type"]["label"], res["sexism_type"]["label"])
                        + "** (this model is weak - treat as a hint).")
            st.write("**Topics found:** " + ", ".join(TOPIC_NAMES[t] for t in res["topics"]))
            terms = res["stance"]["top_terms"]
            if terms:
                st.write("**Words that pushed the stance prediction:** " + ", ".join(terms))
            with st.expander("Cleaned text and probabilities"):
                st.code(res["text_clean"])
                st.json({k: {l: round(p, 3) for l, p in res[k]["probabilities"].items()}
                         for k in ("stance", "toxicity", "sexism")})
        st.caption("Models were trained on tweets, not Instagram, and are imperfect. Use as a guide, not a verdict.")

# ---- Topic insights ------------------------------------------------------------
with tab_topics:
    st.subheader("Where do negative reactions concentrate?")
    st.write("Share of comments that are negative, per topic: sexist (EDOS), hateful or offensive "
             "(Davidson), or against feminism (Stance). A **lift** above 1 means worse than that dataset's average.")
    rates = gold("negative_rate_by_topic")
    source = st.radio("Dataset", ["edos", "stance", "davidson"], horizontal=True,
                      format_func={"edos": "Sexism (EDOS)", "stance": "Feminism stance",
                                   "davidson": "Hate / offensive (Davidson)"}.get)
    view = rates[(rates.source == source) & (rates.topic != "No topic matched")].copy()
    st.bar_chart(view.set_index("topic")["pct_negative"], y_label="% negative")
    st.dataframe(view[["topic", "n", "pct_negative", "pct_negative_overall", "lift", "low_n"]],
                 hide_index=True)
    st.caption("low_n = true means fewer than 30 comments, so do not trust that row.")

    st.subheader("What kind of sexism appears in each topic?")
    types = gold("sexism_type_by_topic")
    types = types[types.topic != "No topic matched"].assign(sub_label=lambda d: d.sub_label.map(lambda v: TYPE_NAMES.get(v, v)))
    st.bar_chart(types.pivot(index="topic", columns="sub_label", values="pct_in_topic").fillna(0),
                 y_label="% of sexist comments in the topic")

    st.subheader("Supportive vs. critical comments per topic (stance dataset)")
    stance = gold("stance_by_topic")
    stance = stance[~stance.low_n & (stance.topic != "No topic matched")].assign(
        label=lambda d: d.label.map(STANCE_NAMES))
    st.bar_chart(stance.pivot(index="topic", columns="label", values="pct_in_topic").fillna(0),
                 y_label="% of comments")

# ---- Recommendations -----------------------------------------------------------
with tab_reco:
    st.subheader("Messaging ideas to test")
    st.warning("These are hypotheses linked to patterns in comments. They are not proven to change anyone's mind "
               "- test them (for example by posting two wordings) and compare the replies.")
    for _, r in gold("recommendations").iterrows():
        with st.expander(f"{r.topic}  -  sexism lift {r.sexism_lift}"):
            st.write("**Suggestion:** " + r.recommendation)
            st.write("**Evidence:** " + r.evidence)

# ---- Model quality -------------------------------------------------------------
with tab_models:
    st.subheader("How good are the models? (held-out test data)")
    summary = gold("model_summary")
    st.dataframe(summary, hide_index=True)
    st.write("Compare **accuracy** with **baseline_accuracy** (always guessing the biggest class). "
             "Toxicity and sexism beat the baseline; sexism type and stance are weak.")
    st.dataframe(gold("class_metrics"), hide_index=True)

# ---- About ---------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
**How it works.** PySpark validates and cleans three public datasets (Davidson hate speech, EDOS sexism,
TweetEval feminism stance), tags topics, trains four text classifiers with cross-validation, and writes small
result files. This app reads those files and scores text in plain Python.

**Limitations (please read).**
- No dataset is from Instagram. All are English tweets, so results may not transfer.
- Topics are found with keyword lists, not human labels. Most comments match no topic.
- Gender-equality keywords overlap with sexist comments, which inflates that topic's lift.
- Stance labels are noisy; the sexism-type and stance models are barely better than guessing.
- No dataset links posts to their comments, so recommendations are hypotheses, not measured effects.
        """
    )