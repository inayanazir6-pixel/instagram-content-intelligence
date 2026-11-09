"""Instagram Content Intelligence - comment triage and insight assistant (no Spark needed).

Reads the small result files made by the Spark pipeline:
  data/gold/*.csv         analysis tables
  models/exports/*.json   trained models, scored with plain Python
Run locally:  streamlit run streamlit_app.py
"""
import html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.reporting import build_report, summarize
from src.scoring import analyze_comment, clean_text, tag_topics
from src.triage import ACTIONS, STANCE_NAMES, Thresholds, decide, read_rows, rows_to_csv, triage_rows

st.set_page_config(page_title="Instagram Content Intelligence", page_icon="📊", layout="wide")

MAX_ROWS = 5000  # keeps the hosted demo fast
REACTION_COLORS = {"Supportive": "#2ECC71", "Neutral": "#9AA0B4", "Critical": "#F5A623", "Toxic": "#E74C3C"}
ACTION_COLORS = {"hide_report": "#E74C3C", "spam": "#D35400", "review": "#F5A623", "question": "#1ABC9C",
                 "reply": "#3498DB", "thank": "#2ECC71", "unsure": "#9B59B6", "ok": "#6B6482"}
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
EXAMPLES = {
    "💚 Supportive": "Girls deserve the same education and pay as boys. Equality benefits everyone.",
    "😐 Neutral": "The panel on women in tech starts at 10am tomorrow in the main hall.",
    "🟠 Critical": "Feminism has gone too far and the movement ignores what men actually go through.",
    "🔴 Sexist": "Women belong in the kitchen, not in the boardroom.",
}
SAMPLE_FILE = config.ROOT / "inbox" / "sample_comments.csv"

# ---- Look and feel -------------------------------------------------------------
st.markdown(
    """
<style>
.hero {background: linear-gradient(120deg,#833AB4 0%,#E1306C 50%,#FCB045 100%);
       padding: 28px 32px; border-radius: 18px; margin-bottom: 18px;}
.hero h1 {color: #fff; margin: 0; font-size: 2.1rem;}
.hero p {color: #fff; opacity: .92; margin: 6px 0 0 0; font-size: 1.02rem;}
.badge {display:inline-block; padding: 10px 22px; border-radius: 999px; color:#fff;
        font-weight: 700; font-size: 1.5rem;}
.pill {display:inline-block; padding: 4px 12px; margin: 2px 6px 2px 0; border-radius: 999px;
       background:#2A2440; border:1px solid #4A3F7A; font-size: .9rem;}
.pill.word {background:#3A2038; border-color:#7A3F6A;}
div[data-testid="stMetric"] {background:#1B1727; padding:12px 16px; border-radius:14px;
                             border:1px solid #2F2947;}
</style>
<div class="hero">
  <h1>📊 Instagram Content Intelligence</h1>
  <p>Sort thousands of comments in seconds: spot abuse, find what deserves a reply, and see which topics attract backlash.</p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def gold(name: str) -> pd.DataFrame:
    df = pd.read_csv(config.GOLD_DIR / f"{name}.csv")
    if "topic" in df:
        df["topic"] = df["topic"].map(lambda v: TOPIC_NAMES.get(v, v))
    return df


@st.cache_data(show_spinner=False)
def run_triage(csv_text: str, limits: tuple):
    fields, rows, text_col = read_rows(csv_text)
    rows = rows[:MAX_ROWS]
    return fields, triage_rows(rows, text_col, Thresholds(*limits)), text_col


def pct(p: float) -> str:
    return f"{100 * p:.0f}%"


def style(fig, height=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, t=40, b=0), height=height, legend_title_text="")
    return fig


def set_example(text: str):
    st.session_state.comment = text


if "comment" not in st.session_state:
    st.session_state.comment = ""

tab_mod, tab_one, tab_caption, tab_insights, tab_quality = st.tabs(
    ["🛡️ Moderation Assistant", "🔍 Single comment", "✍️ Caption check", "📈 Research insights", "🧪 Models & limits"])

# ==== 1. Moderation Assistant =====================================================
with tab_mod:
    st.subheader("Upload your comments, get a sorted action list")
    st.write("Upload a CSV with one comment per row (column named **comment**, text, body or message). "
             "The tool sorts every comment into an action, explains why, and writes a report you can download.")
    up_col, sample_col = st.columns([3, 1])
    uploaded = up_col.file_uploader("CSV file", type="csv", label_visibility="collapsed")
    use_sample = sample_col.button("▶ Try with sample data")
    if use_sample:
        st.session_state.use_sample = True
    if uploaded is not None:
        st.session_state.use_sample = False

    csv_text, source_name = None, ""
    if uploaded is not None:
        csv_text, source_name = uploaded.getvalue().decode("utf-8", errors="replace"), uploaded.name
    elif st.session_state.get("use_sample") and SAMPLE_FILE.exists():
        csv_text, source_name = SAMPLE_FILE.read_text(encoding="utf-8"), "sample_comments.csv (synthetic demo data)"

    with st.expander("⚙️ Sensitivity settings (advanced)"):
        st.caption("Higher numbers flag fewer comments. Lower numbers flag more.")
        c1, c2, c3 = st.columns(3)
        hate = c1.slider("Hate-speech score to hide", 0.3, 0.9, 0.50, 0.05)
        sexist_review = c2.slider("Sexism score to review", 0.4, 0.95, 0.70, 0.05)
        toxic_review = c3.slider("Toxicity score to review", 0.5, 0.99, 0.90, 0.01)
    limits = (hate, 0.85, 0.65, sexist_review, toxic_review, 0.42, 0.60, 0.60)

    if csv_text is None:
        st.info("👆 Upload a file or click **Try with sample data**. Nothing you upload is stored.")
    else:
        try:
            with st.spinner("Analysing comments..."):
                fields, result, text_col = run_triage(csv_text, limits)
        except Exception as error:  # a bad file should never crash the page
            st.error(f"Could not read this file: {error}")
            result = []
        if result:
            s = summarize(result)
            st.caption(f"Source: {source_name} · analysed {s['n']} comments"
                       + (f" (first {MAX_ROWS} only)" if s["n"] >= MAX_ROWS else ""))
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Comments", s["n"])
            k2.metric("Conversation health", f"{s['health']:.0f}%", help="Share of comments needing no moderation")
            k3.metric("Need moderation", s["flagged"])
            k4.metric("Waiting for a reply", s["actions"]["reply"] + s["actions"]["question"])

            left, right = st.columns(2)
            with left:
                donut = pd.DataFrame({
                    "Action": [ACTIONS[k][1] for k in ACTIONS],
                    "Comments": [s["actions"][k] for k in ACTIONS],
                    "key": list(ACTIONS),
                })
                donut = donut[donut.Comments > 0]
                pie = px.pie(donut, names="Action", values="Comments", hole=0.55, title="What needs doing",
                             color="Action", color_discrete_map={ACTIONS[k][1]: ACTION_COLORS[k] for k in ACTIONS})
                st.plotly_chart(style(pie, 340), key="mod_pie")
            with right:
                if s["topics"]:
                    tdf = pd.DataFrame(s["topics"]).assign(topic=lambda d: d.topic.map(lambda v: TOPIC_NAMES.get(v, v)))
                    tb = px.bar(tdf.sort_values("flagged_pct"), x="flagged_pct", y="topic", orientation="h",
                                title="Share of flagged comments by topic", hover_data={"n": True, "flagged": True},
                                labels={"flagged_pct": "% flagged", "topic": ""}, color_discrete_sequence=["#E1306C"])
                    st.plotly_chart(style(tb, 340), key="mod_topics")
                else:
                    st.info("Not enough comments about a single topic yet (need at least 5).")

            st.markdown("#### Action queue")
            labels = {ACTIONS[k][1]: k for k in ACTIONS}
            chosen = st.multiselect("Show", list(labels), default=[ACTIONS[k][1] for k in ("hide_report", "spam", "review", "question", "reply")])
            table = pd.DataFrame(result)
            table = table[table.action.isin([labels[c] for c in chosen])]
            table.insert(0, "Action", table.action.map(lambda k: ACTIONS[k][1]))
            shown = table[["Action", text_col, "reasons", "p_toxic", "p_sexist", "topics", "suggested_reply"]]
            st.dataframe(
                shown, hide_index=True,
                column_config={
                    text_col: st.column_config.TextColumn("Comment", width="large"),
                    "reasons": "Why",
                    "p_toxic": st.column_config.ProgressColumn("Toxic", min_value=0.0, max_value=1.0, format="%.2f"),
                    "p_sexist": st.column_config.ProgressColumn("Sexist", min_value=0.0, max_value=1.0, format="%.2f"),
                    "topics": "Topics", "suggested_reply": st.column_config.TextColumn("Suggested reply", width="large"),
                },
            )

            d1, d2 = st.columns(2)
            report = build_report(result, text_col, title=f"Comment report: {source_name}")
            d1.download_button("⬇️ Download sorted CSV", rows_to_csv(fields, result),
                               file_name="triaged_comments.csv", mime="text/csv")
            d2.download_button("⬇️ Download written report", report, file_name="comment_report.md",
                               mime="text/markdown")
            with st.expander("📄 Preview the written report"):
                st.markdown(report)
            st.caption("An automatic first sort, not a final decision. The models make mistakes, "
                       "so a person should check anything flagged before acting.")

# ==== 2. Single comment ===========================================================
with tab_one:
    st.write("**Try an example or type your own** (English). Results update as you type; press Ctrl+Enter to apply.")
    cols = st.columns(len(EXAMPLES))
    for col, (label, example) in zip(cols, EXAMPLES.items()):
        col.button(label, on_click=set_example, args=(example,))
    text = st.text_area("Comment", key="comment", height=100, label_visibility="collapsed",
                        placeholder="Paste or type a comment here...")

    if text.strip():
        res = analyze_comment(text)
        if res["too_short"]:
            st.warning("Please write at least two words.")
        else:
            d = decide(res, res["text_clean"], Thresholds(), text)
            sx = res["sexism"]
            left, right = st.columns([1, 2])
            with left:
                st.markdown("**Recommended action**")
                st.markdown(f'<span class="badge" style="background:{ACTION_COLORS[d["action"]]}">{ACTIONS[d["action"]][1]}</span>',
                            unsafe_allow_html=True)
                st.caption("Why: " + "; ".join(d["reasons"]))
                st.markdown("**Topics found**")
                st.markdown("".join(f'<span class="pill">{TOPIC_NAMES[t]}</span>' for t in res["topics"]),
                            unsafe_allow_html=True)
                terms = res["stance"]["top_terms"]
                if terms:
                    st.markdown("**Words that pushed the stance**")
                    st.markdown("".join(f'<span class="pill word">{html.escape(t)}</span>' for t in terms),
                                unsafe_allow_html=True)
            with right:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=100 * d["p_hate"], number={"suffix": "%"},
                    title={"text": "Hate-speech score (hide at 50%)"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#E1306C"},
                           "threshold": {"line": {"color": "white", "width": 4}, "value": 50}},
                ))
                st.plotly_chart(style(gauge, 260), key="gauge")

            m1, m2, m3 = st.columns(3)
            m1.metric("Reaction", d["reaction"], d["stance"] + " " + pct(d["stance_conf"]) + " sure", delta_color="off")
            m2.metric("Sexism score", pct(d["p_sexist"]), sx["label"], delta_color="off")
            if "sexism_type" in res:
                label = res["sexism_type"]["label"]
                m3.metric("Likely type of sexism", TYPE_NAMES.get(label, label).split(" (")[0], "weak model - a hint", delta_color="off")
            else:
                m3.metric("Likely type of sexism", "-", "not flagged as sexist", delta_color="off")
            with st.expander("Cleaned text used by the model"):
                st.code(res["text_clean"])
        st.caption("Models were trained on tweets, not Instagram, and are imperfect. Use as a guide, not a verdict.")
    else:
        st.info("👆 Pick an example above or type a comment to see the analysis.")

# ==== 3. Caption check ============================================================
with tab_caption:
    st.subheader("Before you post: which topics does your caption touch?")
    st.write("Paste a draft caption. The tool finds its topics and shows what the research data says about "
             "the replies those topics tend to attract, plus ideas to test.")
    caption = st.text_area("Draft caption", height=110, placeholder="e.g. Equal pay for equal work. Women deserve a fair chance at every job.")
    if caption.strip():
        found = tag_topics(clean_text(caption))
        rates = gold("negative_rate_by_topic")
        edos = rates[rates.source == "edos"].set_index("topic")
        types = gold("sexism_type_by_topic")
        recs = gold("recommendations").set_index("topic")
        if not found:
            st.info("No women's-rights topic was matched, so there is no topic-based advice for this caption.")
        for key in found:
            name = TOPIC_NAMES[key]
            if name not in edos.index:
                continue
            row = edos.loc[name]
            level = "🔴 Higher risk" if row.lift >= 1.3 else ("🟡 Typical" if row.lift >= 1.0 else "🟢 Lower risk")
            with st.container(border=True):
                st.markdown(f"### {name}  ·  {level}")
                a, b, c = st.columns(3)
                a.metric("Sexist comments in this topic", f"{row.pct_negative:.0f}%", f"avg {row.pct_negative_overall:.0f}%", delta_color="off")
                top = types[types.topic == name].sort_values("n", ascending=False)
                if len(top):
                    b.metric("Most common type", TYPE_NAMES.get(top.iloc[0].sub_label, top.iloc[0].sub_label).split(" (")[0])
                c.metric("Comments studied", int(row.n))
                if name in recs.index:
                    st.write("💡 **Idea to test:** " + recs.loc[name].recommendation)
        st.markdown("#### Quick checklist")
        st.markdown("- Add a source link or one clear statistic.\n- Tell a real story instead of a general claim.\n"
                    "- Invite questions so critics can ask instead of attack.\n"
                    "- Post a short comment-rules note, and decide who will moderate.")
        st.caption("Based on patterns in public research data (tweets), not a prediction for your own audience. "
                   "Gender-equality keywords overlap with sexist comments, which inflates that topic's risk.")
    else:
        st.info("👆 Paste a caption to see the topic check.")

# ==== 4. Research insights ========================================================
with tab_insights:
    st.subheader("Where do negative reactions concentrate?")
    st.caption("Negative = sexist (EDOS), hateful or offensive (Davidson), or against feminism (Stance). "
               "Lift above 1 means worse than that dataset's average. Hover over bars for details.")
    rates = gold("negative_rate_by_topic")
    source = st.radio("Dataset", ["edos", "stance", "davidson"], horizontal=True,
                      format_func={"edos": "Sexism (EDOS)", "stance": "Feminism stance",
                                   "davidson": "Hate / offensive (Davidson)"}.get)
    view = rates[(rates.source == source) & (rates.topic != "No topic matched")].copy()
    view["Sample"] = view.low_n.map({True: "Few comments (unreliable)", False: "Enough comments"})
    avg = view.pct_negative_overall.iloc[0]
    fig = px.bar(view.sort_values("pct_negative"), x="pct_negative", y="topic", orientation="h",
                 color="Sample", color_discrete_map={"Enough comments": "#E1306C", "Few comments (unreliable)": "#6B6482"},
                 hover_data={"n": True, "lift": True, "pct_negative": ":.1f", "Sample": False},
                 labels={"pct_negative": "% negative", "topic": ""})
    fig.add_vline(x=avg, line_dash="dash", line_color="white", annotation_text=f"average {avg}%")
    st.plotly_chart(style(fig, 360), key="neg_rate")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Types of sexism by topic")
        types = gold("sexism_type_by_topic")
        types = types[types.topic != "No topic matched"].assign(
            sub_label=lambda d: d.sub_label.map(lambda v: TYPE_NAMES.get(v, v).split(" (")[0]))
        heat = types.pivot(index="topic", columns="sub_label", values="pct_in_topic").fillna(0)
        hm = px.imshow(heat, text_auto=".0f", aspect="auto", color_continuous_scale="Magma",
                       labels={"color": "% of sexist comments"})
        st.plotly_chart(style(hm, 380), key="heatmap")
    with c2:
        st.subheader("Supportive vs. critical (stance)")
        stance_df = gold("stance_by_topic")
        stance_df = stance_df[~stance_df.low_n & (stance_df.topic != "No topic matched")].assign(
            label=lambda d: d.label.map(STANCE_NAMES))
        sb = px.bar(stance_df, x="topic", y="pct_in_topic", color="label", barmode="group",
                    color_discrete_map=REACTION_COLORS, labels={"pct_in_topic": "% of comments", "topic": ""})
        st.plotly_chart(style(sb, 380), key="stance_topic")
        st.caption("Topics with fewer than 30 comments are hidden.")

    st.subheader("Messaging ideas to test")
    st.warning("These are hypotheses linked to patterns in comments. They are not proven to change anyone's mind. "
               "Test them (for example by posting two wordings) and compare the replies.")
    left, right = st.columns(2)
    for i, (_, r) in enumerate(gold("recommendations").iterrows()):
        with (left if i % 2 == 0 else right):
            with st.container(border=True):
                st.markdown(f"**{r.topic}**")
                st.markdown(f'<span class="pill">sexism lift {r.sexism_lift}</span>'
                            f'<span class="pill">main type: {TYPE_NAMES.get(r.dominant_sexism_type, r.dominant_sexism_type).split(" (")[0]}</span>',
                            unsafe_allow_html=True)
                st.write("💡 " + r.recommendation)
                st.caption("Evidence: " + r.evidence)

# ==== 5. Models & limits ==========================================================
with tab_quality:
    st.subheader("How good are the models? (held-out test data)")
    summary = gold("model_summary")
    long = summary.melt(id_vars="task", value_vars=["accuracy", "baseline_accuracy"],
                        var_name="measure", value_name="score")
    long["measure"] = long.measure.map({"accuracy": "Model", "baseline_accuracy": "Always guess the biggest class"})
    qb = px.bar(long, x="task", y="score", color="measure", barmode="group", range_y=[0, 1],
                color_discrete_map={"Model": "#E1306C", "Always guess the biggest class": "#6B6482"},
                labels={"task": "", "score": "Accuracy"})
    st.plotly_chart(style(qb, 360), key="quality")
    st.write("Toxicity and sexism beat the baseline. Sexism type and stance are weak, so treat them as hints.")
    metrics = gold("class_metrics")
    task = st.selectbox("Per-class results for", metrics.task.unique())
    st.dataframe(
        metrics[metrics.task == task].drop(columns="task"), hide_index=True,
        column_config={c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.2f")
                       for c in ("precision", "recall", "f1")},
    )
    st.markdown(
        """
**How it works.** PySpark validates and cleans three public datasets (Davidson hate speech, EDOS sexism,
TweetEval feminism stance), tags topics, trains four text classifiers with cross-validation, and writes small
result files. This app reads those files and scores text in plain Python. A parity check confirms it gives
the same answers as Spark. A GitHub automation also triages any CSV dropped into the `inbox/` folder.

**Why triage is built this way.** The models are imperfect, so the tool combines them with simple word rules
(threats, spam, insults, sexist phrases, questions), like real moderation tools do. Hiding a comment needs strong
evidence, unclear comments go to a person, and every action shows its reason.

**Limitations (please read).**
- No dataset is from Instagram. All are English tweets, so results may not transfer.
- Topics are found with keyword lists, not human labels. Most comments match no topic.
- Gender-equality keywords overlap with sexist comments, which inflates that topic's lift.
- Stance labels are noisy; the sexism-type and stance models are barely better than guessing.
- Caption advice comes from topic-level patterns, not from tests on real posts.
        """
    )
