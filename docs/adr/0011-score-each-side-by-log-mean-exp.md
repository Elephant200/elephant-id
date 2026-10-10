# Score each side by the log-mean-exp of its ear similarities

**Status:** builds on [ADR 0004](0004-treat-the-catalog-as-per-side-evidence.md)

The left score of a candidate is the log-mean-exp of the ear similarities of all its left reference ears, `τ * (logsumexp(s / τ) - log n)`. The right score is the same for the right ears. The two sides stay independent.

The log-mean-exp is a smooth maximum. A candidate scores about its best ear similarity, minus `τ * log(n / k)` when only `k` of its `n` references agree. The alternatives have these problems:

- The maximum uses one reference only. It ignores agreement from the other sightings of a known elephant, and it gives a candidate with many references more chances at one lucky high match.
- The softmax-weighted mean ties with the log-mean-exp on the tuning set, but it is not monotone: a better match with a weak reference can lower the score. The log-mean-exp is monotone.
- The top-k mean favours elephants with many sightings even more than the maximum does.

A fixed `τ` gives the candidate score the same meaning in every catalog, and each candidate score uses only the query and that candidate's references.
