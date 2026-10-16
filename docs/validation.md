# Public release verification

Validation date: 2026-10-16. This record concerns the public synthetic package, not the private historical experiment.

| Check | Observed result |
|---|---|
| Python | 3.12.14, Windows x64 |
| Complete public pytest suite | 307 passed; 0 failed; 0 errors; 0 skipped |
| Warning policy | Warnings treated as errors; none reported in the passing run |
| `pip check` in the validation runtime | No broken requirements found |
| Offline synthetic demo | Causal feature perturbation, missing label tail, next-session execution and dividend NAV conservation passed |
| Original source equivalence | 105 source modules copied without research-code changes |
| Publication scan | No detected credentials, personal home paths, non-noreply email addresses or excluded artifacts |

The demonstration's deterministic accounting fingerprint is:

```text
5c63400b9b7f1554fc4bc32c078fefad805efbd8c7746236b57fa94748ae8480
```

Tests use generated observations and literal accounting examples. Some tests use synthetic future dates to verify adapter behavior; these are not real final-window data or results. The public suite denies socket connections.

The initial demo used an unsupported order-source string; the engine correctly refused it. The demo was changed to the already-supported `manual_test` input. No engine rule was relaxed. The full public suite was then rerun successfully.

The passing local run used the existing research runtime. The first hosted Ubuntu CI run installed dependencies and passed `pip check`, but reported 293 passed and 14 failed: the public requirements omitted AKShare metadata needed by snapshot manifests, even when fetches are synthetic. The requirements now include the original installed AKShare version (1.19.1); no test or research logic was weakened. The workflow history records the failed run and subsequent verification. Hosted results must be checked at the exact commit being reviewed. The public requirements pin the direct/test dependencies, not every transitive dependency or operating-system library.

This release does not claim that private raw data, trained models or historical account publications can be reproduced from public files alone. Private pipeline-wide test counts are deliberately not used here.
