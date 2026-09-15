---
name: mobilint-sdk-tutorial
description: Apply the Mobilint SDK tutorial repository workflow when editing its documentation, example scripts, local helpers, bilingual README pairs, or hardware-dependent validation guidance.
---

# Mobilint SDK Tutorial Skill

Before starting work, read and follow the canonical shared guide at
[`.agents/agent-guide.md`](../../../.agents/agent-guide.md). It is the single
source for both general agent instructions and this skill's workflow.

## Synchronization Policy

`.agents/agent-guide.md` is canonical. `AGENTS.md` is the repository entrypoint
and `CLAUDE.md` is a symlink to it. This file stays a thin skill entrypoint
that points at the canonical guide; it keeps its own frontmatter because Claude
Code skill discovery requires a `name` field.

A major workflow change requires updating the canonical guide and any affected
entrypoint in the same change. Major workflow changes include changes to the
repository map, tutorial architecture, SDK or tooling setup, validation
process, dependency expectations, documentation policy, or this entrypoint
layout; ordinary tutorial-content edits are not major workflow changes.
