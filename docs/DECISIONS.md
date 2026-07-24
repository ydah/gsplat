# Architecture Decision Records

Architecture Decision Records (ADRs) capture significant technical decisions, their context, and
their consequences. The individual records live in [`docs/decisions/`](decisions/) and are never
deleted; a later ADR supersedes an earlier decision when the architecture changes.

This structure follows the concise format used by
[ydah/ibex](https://github.com/ydah/ibex/tree/main/docs/decisions).

## Index

| ADR | Status | Date | Decision |
|---|---|---|---|
| [0001](decisions/0001-pin-golden-reference-and-cpu-validation.md) | Accepted | 2026-07-23 | Pin the golden reference and define CPU-only validation |
| [0002](decisions/0002-retain-ruby-fallbacks-for-native-backend.md) | Accepted | 2026-07-23 | Retain exact Ruby fallbacks for the native backend |
| [0003](decisions/0003-share-projection-for-distorted-cameras.md) | Accepted | 2026-07-23 | Share the Ruby projection for distorted cameras |
| [0004](decisions/0004-use-portable-world-space-reference-paths.md) | Accepted | 2026-07-24 | Use one portable reference for world-space extension paths |
| [0005](decisions/0005-reuse-ewa-core-for-2dgs.md) | Accepted | 2026-07-24 | Reuse the differentiable EWA core for 2DGS |
| [0006](decisions/0006-keep-eval3d-as-portable-reference.md) | Accepted | 2026-07-24 | Keep eval3d as a portable reference path |
| [0007](decisions/0007-share-compositor-semantics-for-contribution-indices.md) | Accepted | 2026-07-24 | Share compositor semantics for contribution indices |
## When an ADR is required

Write an ADR when a change introduces a meaningful trade-off or establishes a long-lived contract,
including:

- public API, serialized format, or compatibility policy changes;
- backend ownership, fallback, precision, or performance boundaries;
- algorithm choices whose rejected alternatives may be reconsidered;
- validation policy changes or accepted differences from the upstream implementation;
- decisions that supersede or deprecate an existing ADR.

Routine implementation details that follow an accepted decision do not need another ADR.

## Creating an ADR

1. Copy [`0000-template.md`](decisions/0000-template.md).
2. Use the next four-digit number and a lowercase kebab-case filename:
   `NNNN-short-decision-title.md`.
3. Keep the title number equal to the filename number and use an ISO `YYYY-MM-DD` date.
4. Add the ADR to the index in the same change.
5. Run `bundle exec ruby -Itest test/architecture_decision_records_test.rb`.

## Status lifecycle

- `Proposed`: under review and not yet authoritative.
- `Accepted`: active architecture.
- `Rejected`: considered but not adopted.
- `Deprecated`: retained for history but no longer recommended.
- `Superseded by NNNN`: replaced by a newer ADR.

Do not rewrite an accepted ADR to represent a different decision. Add a new ADR, mark the old one
`Superseded by NNNN`, and link the records from their context or consequences.
