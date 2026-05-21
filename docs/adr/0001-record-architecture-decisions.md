# ADR 0001 - Record architecture decisions

**Status:** accepted - 2026-05-19

## Context
The repository evolves over months and is itself an artifact for a PhD application.
Significant design decisions should be traceable.

## Decision
Record every significant, hard-to-reverse decision as a short ADR in `docs/adr/`,
numbered sequentially. ADR-worthy examples: choosing SB3 versus a custom trainer, the
config schema, the encoder interface, the navigation task set.

## Consequences
A small per-decision overhead, in exchange for a codebase whose reasoning stays legible
to reviewers and to future-you.
