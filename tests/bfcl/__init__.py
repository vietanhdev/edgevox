"""BFCL v3-style tool-calling regression harness.

The Berkeley Function Calling Leaderboard (BFCL) v3 evaluates a model's
ability to emit syntactically- and semantically-correct function calls
across categories: simple, parallel, multiple, multi-turn, irrelevance,
relevance.

This package mirrors the *evaluation* surface — the AST-equality
predicate that compares two tool-call lists for equivalence — so we can
regression-test the parser chain + grammar decoding without pulling in
the upstream dataset.

Full upstream fixture downloads are gated behind ``pytest.mark.bfcl_full``
so CI stays offline + fast. Use ``pytest -m bfcl_full`` after running
``scripts/fetch_bfcl_v3.py`` (when shipped) to run the full battery.
"""
