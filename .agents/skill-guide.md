---
description: Canonical workflow for aligned documentation and example-script
  changes in the Mobilint SDK tutorial repository.
paths:
  - "**"
---

# Mobilint SDK Tutorial Skill Guide

Use this repository as a documentation-first tutorial workspace. Keep example
scripts, README commands, and expected artifacts synchronized within each
tutorial directory.

## Synchronization Policy

Keep the Codex and Claude agent guides aligned, and keep shared skill guidance
and its tool-specific entrypoints aligned. A major workflow change requires
reviewing and updating all applicable shared guidance and entrypoints in the
same change. Major workflow changes include changes to the repository map,
tutorial architecture, SDK or tooling setup, validation process, dependency
expectations, or documentation policy; ordinary tutorial-content edits are not
major workflow changes.

The shared guides in `.agents/` are canonical. Keep `AGENTS.md`, `CLAUDE.md`,
and `.claude/skills/mobilint-sdk-tutorial/SKILL.md` as tool-specific entrypoints
that point to the applicable canonical guide.

## First Reads

Start with the smallest set of files that anchor the task:

- `README.md` and `README.KR.md` for the top-level product framing
- The nearest `compilation/<task>/README.md` or `runtime/<task>/README.md`
- `runtime/_guides/*.md` when the task affects shared runtime concepts or
  terminology
- The adjacent script files used by that tutorial
- `pyproject.toml` for Python version and Ruff settings
- `package.json` when Markdown validation or docs tooling matters
- `git status --short` before editing so you do not overwrite unrelated user
  work

If the touched tutorial has a Korean counterpart, open `README.KR.md` early so
structure changes can stay aligned.

## Current Repo State

- The repo contains paired compilation and runtime tutorials under
  `compilation/` and `runtime/`.
- Many tutorials have both `README.md` and `README.KR.md`; where both exist,
  update their structure, commands, paths, defaults, and user-visible behavior
  together. Do not create a missing Korean mirror unless the task explicitly
  requests it.
- Runtime Python tutorials commonly use small local helper modules such as
  `utils.py`, `postprocess.py`, `visualize.py`, and dataset label files like
  `coco.py` or `dota.py`.
- Some runtime Python tutorials use local wrappers instead of a single direct
  `qbruntime` script, notably `runtime/python/bert/` and
  `runtime/python/llm/`.
- Runtime STT and VLM Python tutorials currently focus on `mblt-model-zoo`
  style flows rather than a direct local `qbruntime` inference script.
- Runtime C++ tutorials include local `utils/inference/` and
  `utils/postprocess/` helpers, so README changes there often need code checks
  beyond the top-level `infer_*.cc`.
- The workspace includes generated `tmp/` and `__pycache__/` directories in
  tutorial folders; do not treat them as authored source.
- `compilation/oriented_bounding_boxes/` contains the `YOLO11m-obb`
  compilation flow that produces `yolo11m-obb.mxq`.
- `runtime/python/oriented_bounding_boxes/` is a self-contained OBB runtime
  tutorial that expects
  `../../../compilation/oriented_bounding_boxes/yolo11m-obb.mxq`, uses
  `1024x1024` letterbox preprocessing, keeps runtime input as `uint8`, decodes
  DOTA OBB rows as `cx, cy, w, h, conf, cls, angle`, and renders rotated
  polygons.

## Repo Map

- `compilation/README.md`: Compiler setup, Docker workflow, and qbcompiler
  installation guidance
- `runtime/README.md`: Runtime setup, driver/library installation, and NPU
  assumptions
- `runtime/_guides/*.md`: Shared conceptual docs referenced by the runtime
  overview
- `compilation/*`: Model-specific compilation walkthroughs
- `runtime/python/*`: Python runtime walkthroughs
- `runtime/cpp/*`: C++ runtime walkthroughs where present
- `assets/`: Shared diagrams and logo assets
- Per-directory helper modules such as `imagenet.py`, `coco.py`, `dota.py`,
  `postprocess.py`, `utils.py`, `visualize.py`, and `wrapper/`: Local helpers
  for one tutorial family

## Working Rules

- Treat each tutorial as self-contained unless there is already a local shared
  helper in that area.
- When you change a script interface, update the matching README in the same
  directory.
- Prefer concrete filenames and copy-pasteable commands.
- Where `README.md` and `README.KR.md` both exist, keep their structure,
  commands, paths, defaults, and user-visible behavior synchronized. Do not
  create a missing translation unless the task explicitly requests it.
- Prefer direct, readable scripts that mirror the tutorial text instead of
  introducing shared library-style abstractions.
- Preserve the tutorial's existing execution style. Do not collapse
  wrapper-based or `mblt-model-zoo` examples into direct `qbruntime` calls
  unless the task explicitly requires it.
- Keep README sections ordered around user workflow: prerequisites,
  preparation, execution, output.
- Reflect external constraints explicitly when examples depend on Mobilint
  proprietary wheels, Mobilint NPU devices, Docker images, Hugging Face
  authentication, or large downloads.
- For hardware-dependent examples, do static validation when full execution is
  not available and state that limit clearly.

## Validation Defaults

For touched Python tutorial files, prefer:

```bash
ruff check path/to/file.py
ruff format path/to/file.py
python -m py_compile path/to/file.py
```

For a small tutorial directory with several edited scripts:

```bash
python -m compileall path/to/tutorial_dir
```

For touched docs, verify that links, paths, commands, and default filenames
match the local scripts. When useful, run `npx markdownlint path/to/file.md`
because this repo includes Markdown lint tooling when `package.json` is
present.
