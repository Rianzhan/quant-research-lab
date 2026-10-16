# Engineering evidence and reviewer guide

## A 10-minute review

1. Run the offline public test suite and synthetic demo.
2. Inspect `test_future_prices_or_truncation_never_change_past_features`: later prices are perturbed, earlier features must remain identical.
3. Inspect the endpoint purge test: an unusually long label invalidates an earlier row, demonstrating why positional deletion is inadequate.
4. Inspect the dividend test: registration, ex-date receivable and payment date have different economic meanings.
5. Inspect the delayed corporate-action tests: unavailable terms cannot be backfilled into earlier decisions.
6. Inspect order sizing and missing-open tests: a next-session price can reduce an affordable fill but cannot rewrite close-time intent.

## Skills demonstrated

| Capability | Concrete evidence |
|---|---|
| Research question to explicit contracts | Shared feature whitelist, fold membership, frozen strategy definitions |
| Temporal reasoning | Label endpoint purge, known-at/effective-date separation, next-session execution |
| Experimental control | Same test keys across model scopes; fixed costs and rules before strategy P&L |
| Numerical engineering | Decimal accounting, receivables, split handling, reconciliation and undefined metric cases |
| Reproducibility | Deterministic sorting, fingerprints, immutable publication manifests and source hashes |
| Failure handling | Fail-closed inputs, atomic snapshots, no fake fills or silent provider substitution |
| Scientific communication | Weak-result interpretation and explicit downgrade of holdout identity |

## AI-assisted development disclosure

The project was developed collaboratively with AI coding assistance. Human involvement included choosing the question and constraints, authorizing stages, accepting/rejecting research conventions, freezing protocols, and adjudicating the exposure incident. The repository does not claim unaided authorship or independently prove mastery of every implementation detail.

For an application or interview, the author should be able to explain a test, trace an order through the ledger, and describe a limitation in their own words. Suggested prompts:

- Why is a five-session prediction target different from the return on a next-open trade?
- Why can a perfectly shift-aligned label still leak across a fold boundary?
- Why does a dividend payment not create NAV, and why is a receivable not spendable cash?
- How can label-dependent prediction coverage make a historical strategy appear to hold cash?
- What remains testable after an outcome-exposure incident?

## Scope of reproducibility

The public suite uses artificial inputs and tests actual accepted modules. It does not reproduce private provider snapshots, frozen fitted models, or the full historical account publications. No private-suite pass count is substituted for the public-suite result. Public verification is recorded separately in `docs/validation.md` and CI.
