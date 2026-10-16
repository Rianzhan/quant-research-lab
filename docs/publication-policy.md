# Public release boundary

This is a new public snapshot, not a push of the original repository or its history.

## Included

- Accepted research source modules through the pre-unseal implementation stage.
- A selected, independently runnable suite of synthetic unit and integration tests.
- Reviewed configuration contracts containing public ETF identifiers and research assumptions.
- A synthetic demonstration, methodology notes, and a publication safety check.

Source modules are copied without changing their research logic. Git attributes preserve exact file bytes for the public SHA-256 manifest. The synthetic test helper removes an unused private-manifest fixture; public tests add a network-denial fixture. Public documentation and packaging are new. In-progress event-audit code is excluded.

## Excluded

- All raw/processed market observations, including the sealed 2026 payload and mixed-year files.
- Predictions, labels, fitted estimators, scalers, account ledgers and private performance outputs.
- Archived third-party announcements/PDFs, source-page captures and notebooks with outputs.
- Credentials, environment files, tokens, private keys, local usernames and absolute home paths.
- Chat histories, internal incident snippets, operational logs, caches, virtual environments and the original `.git` directory.

Public source code may refer to absent research manifests and hashes. References are not the corresponding data. Frozen replay entry points must fail closed when those files are absent; do not fabricate files or disable checks to make them run.

Stage-specific configuration files preserve their historical contents. In particular, the original Phase 6A execution configuration predates the accepted correction to one split's announcement timing. The accepted Phase 6A.5 engine is included, but the private evidence overlay is not. These historical configurations are review material, not a ready-to-run live or final-window configuration. The public demo supplies explicit synthetic events instead.

Market-data and third-party-document redistribution rights have not been established here. Their absence is deliberate. No broad open-source license is added on the owner's behalf; public visibility does not grant an unrestricted reuse license.

`tools/check_publication.py` checks the actual public file set for forbidden artifacts, common credential patterns and personal paths. It is a release guard, not a guarantee that a heuristic scanner can find every possible secret. A local provenance audit also checks copied source bytes against the original read-only project.

The original research workspace and its frozen publications are not modified by this export.
