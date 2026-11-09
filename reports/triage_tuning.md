# Triage settings tried on the EDOS test split

3000 comments, 733 truly sexist. Ranked by F1 (balance of recall and precision).

| Hate score | Sexism score | Toxicity score | Flagged | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| 0.8 | 0.5 | 0.95 | 34% | 62% | 44% | 52 |
| 0.9 | 0.5 | 0.95 | 34% | 61% | 45% | 52 |
| 0.8 | 0.5 | 0.9 | 39% | 67% | 42% | 51 |
| 0.9 | 0.5 | 0.9 | 39% | 66% | 42% | 51 |
| 0.7 | 0.5 | 0.95 | 35% | 62% | 44% | 51 |
| 0.7 | 0.5 | 0.9 | 40% | 67% | 41% | 51 |
| 0.8 | 0.5 | 0.98 | 31% | 58% | 45% | 51 |
| 0.6 | 0.5 | 0.95 | 36% | 63% | 43% | 51 |

**Current settings** (0.5 / 0.7 / 0.9): flagged 34%, recall 54%, precision 39%, F1 45.

_EDOS labels only sexism, so this favours settings that find sexism. Hateful or abusive comments that are not sexist count as mistakes here._
