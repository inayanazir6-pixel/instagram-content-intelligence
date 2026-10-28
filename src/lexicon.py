r"""Keyword lists (regex fragments) for the six women's-rights topics.

They are matched against `text_clean` (lower-case, no punctuation). `\w*` lets a
stem match several word forms, e.g. `educat\w*` matches education, educated, educate.
These topics are RULE-BASED labels made by us, not human annotations: say so in the report.
"""

TOPIC_KEYWORDS = {
    "gender_equality": [
        r"equalit\w*", r"equal rights", r"equal pay", r"gender gap", r"feminis\w*",
        r"patriarch\w*", r"misogyn\w*", r"sexis\w*", r"women'?s rights", r"gender",
    ],
    "womens_education": [
        r"educat\w*", r"school\w*", r"universit\w*", r"college", r"literacy",
        r"student\w*", r"scholarship\w*", r"learn\w*", r"teacher\w*",
    ],
    "employment": [
        r"jobs?", r"career\w*", r"workplace", r"salar\w*", r"wages?", r"employ\w*",
        r"hired", r"hiring", r"promotion\w*", r"boss", r"working (?:women|woman|mothers?)",
        r"pay gap", r"glass ceiling", r"ceo",
    ],
    "independence": [
        r"independen\w*", r"empower\w*", r"freedom", r"autonomy", r"my choice",
        r"free to choose", r"financial\w*", r"self reliant", r"own money", r"strong women",
    ],
    "harassment_violence": [
        r"harass\w*", r"rape\w*", r"rapist\w*", r"assault\w*", r"abus\w*", r"violen\w*",
        r"molest\w*", r"stalk\w*", r"metoo", r"domestic", r"victim\w*", r"catcall\w*",
        r"groped", r"consent",
    ],
    "traditional_roles": [
        r"kitchen", r"housewi\w*", r"stay at home", r"stay home", r"homemaker\w*",
        r"sandwich", r"cook\w*", r"household", r"submissive", r"obey\w*", r"traditional",
        r"real man", r"wives", r"wife", r"housework", r"motherhood",
    ],
}