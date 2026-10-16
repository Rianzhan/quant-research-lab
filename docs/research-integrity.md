# Research integrity and evaluation status

The development pipeline was frozen after a Level 1 interpretation: weak or localized historical economic evidence with insufficient stability. Final model families, features, parameters, thresholds, allocations and evaluation rules were not reopened to improve that conclusion.

## exposure-001

During official calendar/corporate-action research, a search service unexpectedly returned target-related outcome snippets. The local sealed price files were not opened, and final-window predictions, orders, NAV and performance were not produced. Nevertheless, external outcome exposure breaks the claim that the final window is completely unseen.

The human investigator therefore permanently recorded:

```json
{
  "statistical_identity": "protocol_locked_non_pristine_holdout_evaluation",
  "final_holdout_pristine": false,
  "external_2026_information_exposure": true,
  "target_related_exposure": true,
  "model_contaminated": false,
  "strategy_rules_contaminated": false,
  "evaluation_rules_contaminated": false
}
```

The three `contaminated=false` fields record that the already-frozen artifacts and rules were not changed using the exposure; they do **not** restore blindness or remove possible interpretive bias. The exposed values and private conversation are intentionally not republished.

Any later authorized evaluation must retain the original H-U/H0/H1/H2 interpretation rules and report their mechanical outcome as a **preregistered protocol outcome**. It must not be described as pristine final-holdout confirmation. Assets, dates and windows are not replaced to recover an appearance of unseen evidence.

As of this public snapshot, the official event audit is paused and no final-window evaluation has been executed. The next genuinely unseen test must be prospective, not a convenient historical window selected after exposure.

## Why this belongs in a portfolio

The ability to recognize and document a broken assumption is part of research quality. A transparent limitation is stronger evidence of scientific judgment than an unsupported success claim. This release makes the engineering reviewable while clearly separating reviewable synthetic invariants from non-public empirical results.
