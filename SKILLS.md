---
name: mblt-sdk-tutorial
description: Use this skill when working in the Mobilint SDK tutorial repository. It covers the documentation-first tutorial layout, paired compilation and runtime examples, bilingual README maintenance, and safe validation in environments that may lack Mobilint hardware or proprietary SDK dependencies.
paths:
  - "**"
---

# Mobilint SDK Tutorial

## When to Use This Skill

Use this skill for changes anywhere in this repository, especially when the task involves:

- Tutorial docs under `README.md`, `README.KR.md`, `compilation/`, or `runtime/`
- Standalone example scripts that accompany a specific tutorial
- Keeping script arguments, filenames, and README commands synchronized
- Updating bilingual documentation where English and Korean versions coexist
- Validation planning for workflows that depend on `qbcompiler`, `qbruntime`, Docker, NPU devices,
  gated datasets, or large model downloads

## First Reads

Start with the smallest set of files that anchor the task:

- `README.md` and `README.KR.md` for the top-level product framing
- The nearest `compilation/<task>/README.md` or `runtime/<task>/README.md`
- The adjacent script files used by that tutorial
- `pyproject.toml` for Python version and Ruff settings
- `git status --short` before editing so you do not overwrite unrelated user work

If the touched tutorial has a Korean counterpart, open `README.KR.md` early so structure changes
can stay aligned.

## Repo Map

- `compilation/README.md`: Compiler setup, Docker workflow, and qbcompiler installation guidance
- `runtime/README.md`: Runtime setup, driver/library installation, and NPU assumptions
- `compilation/image_classification`, `compilation/object_detection`,
  `compilation/instance_segmentation`, `compilation/pose_estimation`, `compilation/bert`,
  `compilation/llm`, `compilation/stt`, `compilation/tts`, `compilation/vlm`,
  `compilation/face_detection`: Model-specific compilation walkthroughs
- `runtime/image_classification`, `runtime/object_detection`, `runtime/instance_segmentation`,
  `runtime/pose_estimation`, `runtime/bert`, `runtime/llm`, `runtime/stt`, `runtime/tts`,
  `runtime/vlm`, `runtime/face_detection`: Model-specific runtime walkthroughs
- `assets/`: Shared diagrams and logo assets
- Per-directory helper modules such as `imagenet.py`, `coco.py`, `postprocess.py`, `utils.py`,
  and `visualize.py`: Local helpers for one tutorial family

## Working Conventions

### Tutorial Structure

This repo is organized around self-contained examples rather than reusable packages.

- Keep logic local to the example unless a sibling helper module already exists.
- Prefer direct, readable scripts that mirror the tutorial text.
- Keep README sections ordered around user workflow: prerequisites, preparation, execution, output.

### Script and README Sync

When editing code in a tutorial directory:

- Update the matching README command examples and parameter descriptions.
- Keep default argument values aligned with documented example commands.
- Preserve obvious learning landmarks such as “prepare model”, “prepare calibration data”, “compile
  model”, and “run inference”.
- Avoid adding abstractions that make the example harder for a new SDK user to follow.

### Bilingual Docs

Many tutorials ship with both English and Korean READMEs.

- Update both language versions when both exist and the change affects commands, structure,
  filenames, prerequisites, or behavior.
- Keep headings and code blocks parallel across the two files.
- If a tutorial currently lacks `README.KR.md`, do not create one unless the task asks for it.

### Dependency and Environment Assumptions

Examples may depend on:

- Mobilint proprietary wheels such as `qbcompiler`
- Mobilint runtime packages such as `mobilint-qb-runtime`
- Mobilint NPU device nodes like `/dev/aries0`
- Docker images from Mobilint
- Hugging Face authentication and gated datasets
- Model downloads from Torch, Transformers, or the Mobilint model zoo

Reflect these constraints explicitly in docs and avoid implying the example is fully self-contained
when it is not.

## Validation

Use the narrowest validation that is honest for the environment.

### Markdown and Doc Validation

- Check that referenced files, directories, and commands exist.
- Reconcile README examples with script defaults and option names.
- Keep relative links valid after edits.

### Python Validation

For touched scripts, prefer:

```bash
ruff check path/to/file.py
ruff format path/to/file.py
python -m py_compile path/to/file.py
```

For a small tutorial directory with several edited scripts:

```bash
python -m compileall path/to/tutorial_dir
```

### Hardware-Limited Situations

- Do not run broad compilation or inference commands unless the needed hardware and SDK pieces are
  installed.
- If execution requires NPU access, Docker, proprietary wheels, or authenticated dataset downloads,
  call that out in the final report.
- Prefer static verification over speculative fixes that claim runtime success.

## Things to Avoid

- Do not convert the repo into a shared library architecture.
- Do not introduce repo-wide dependencies for a single tutorial unless clearly justified.
- Do not leave README instructions inconsistent with script behavior.
- Do not silently ignore the Korean counterpart when the English doc changed materially.
- Do not revert unrelated local changes already present in the tree.
