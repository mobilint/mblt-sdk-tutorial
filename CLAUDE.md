---
description: Entrypoint for agents working in the Mobilint SDK tutorial repository.
paths:
  - "**"
---

# Agent Instructions

Before starting work, read and follow the canonical shared guide at
[`.agents/agent-guide.md`](.agents/agent-guide.md).

## Synchronization Policy

`.agents/agent-guide.md` is the single canonical guide, covering both general
agent instructions and the `mobilint-sdk-tutorial` skill workflow. This file is
the repository entrypoint and `CLAUDE.md` is a symlink to it, so there is no
separate Codex copy and Claude copy to keep aligned.
`.claude/skills/mobilint-sdk-tutorial/SKILL.md` stays a thin skill entrypoint
that points at the canonical guide.

A major workflow change requires updating the canonical guide and any affected
entrypoint in the same change. Major workflow changes include changes to the
repository map, tutorial architecture, SDK or tooling setup, validation
process, dependency expectations, documentation policy, or this entrypoint
layout; ordinary tutorial-content edits are not major workflow changes.
