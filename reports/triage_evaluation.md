# Triage evaluation on the EDOS test split

3000 comments never seen in training. 733 (24%) are truly sexist.

| Measure | Result |
|---|---|
| Comments flagged (hide / spam / review) | 1024 (34%) |
| Truly sexist comments that were flagged (recall) | 396 of 733 (54%) |
| Flagged comments that are truly sexist (precision) | 396 of 1024 (39%) |

## Which part of the tool flagged what

| Part | Flagged | Truly sexist |
|---|---|---|
| toxicity model alone | 293 | 97 (33%) |
| hate-speech model | 238 | 56 (24%) |
| sexism model alone | 163 | 96 (59%) |
| toxicity + sexism models agree | 145 | 105 (72%) |
| insult word rule | 139 | 28 (20%) |
| threat rule | 42 | 12 (29%) |
| sexist phrase rule | 4 | 2 (50%) |

_Flagged comments that are not labelled sexist may still be rude or hateful, so precision here is a strict, low estimate._
