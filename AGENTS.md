---
description: Repository-wide instructions for agents working in the Mobilint SDK tutorial repository.
paths:
  - "**"
---

# Agent Guidelines

## Scope

These instructions apply across the repository. This repo is a documentation-first tutorial
workspace for Mobilint SDK qb, not a single importable Python package. Most changes will touch
Markdown guides, example scripts, or both.

## Repo Map

- `README.md`, `README.KR.md`: Top-level overview for the full tutorial repo
- `compilation/`: Tutorials and helper scripts for qbcompiler workflows
- `runtime/`: Tutorials and helper scripts for qbruntime workflows
- `assets/`: Shared images used by the docs
- `compilation/*/README.md`, `runtime/*/README.md`: English task-specific guides
- `compilation/*/README.KR.md`, `runtime/*/README.KR.md`: Korean counterparts when available
- `compilation/*/*.py`, `runtime/*/*.py`: Standalone example scripts used by the guides
- `*/**/requirements.txt`: Per-tutorial dependency hints for heavier examples such as LLM, STT,
  and VLM flows

## Working Model

- Treat each tutorial directory as a self-contained example.
- Keep commands, filenames, and argument defaults aligned between the README and the scripts it
  describes.
- Prefer small, explicit scripts over shared abstractions unless the repo already has a local
  utility module for that example.
- Assume users follow the docs step by step on Linux or in Docker, often with Mobilint NPU access.

## Documentation Rules

### Keep Tutorials Operational

- When you change a script interface, update the matching README in the same directory.
- Preserve the beginner-friendly, instructional tone of the existing docs.
- Keep command examples copy-pasteable.
- Prefer concrete file names such as `resnet50.onnx` or `resnet50.mxq` over abstract placeholders
  when the example is fixed to one model.

### Bilingual Content

- Many tutorials exist in both English and Korean.
- If you edit a tutorial with both `README.md` and `README.KR.md`, update both unless the user
  explicitly asks for a single-language change.
- Some directories currently only have English docs. Do not invent missing Korean mirrors unless
  the task calls for it.
- Keep structure, headings, filenames, and commands synchronized across language versions even if
  the prose is not a literal translation.

### Markdown Style

- Use ATX headers (`#`, `##`, `###`).
- Keep a blank line around headings, lists, and fenced code blocks.
- Always include a language tag on fenced code blocks.
- Use inline code for commands, paths, package names, flags, and filenames.
- Use hyphens for unordered lists.
- Preserve existing HTML blocks used for centered images and markdownlint directives.
- Use descriptive alt text for images.
- When a Markdown file serves as an agent rule or reusable workflow, include YAML frontmatter with
  `description` and `paths`.

### Link Hygiene

- Prefer relative repository links in Markdown.
- Verify links after refactors. Some docs may contain stale paths; if you touch them, fix the link
  rather than preserving known breakage.
- Keep external links pointed at official Mobilint, Docker, NVIDIA, Hugging Face, PyTorch, or
  other primary documentation sources.

## Python Script Guidelines

### General Style

- Target Python 3.10 or later, matching `pyproject.toml`.
- Use 4 spaces for indentation.
- Keep lines at 120 characters or fewer when practical.
- Use Ruff formatting and import ordering for new or modified Python files.
- Add type hints to new or significantly edited function signatures when it improves clarity.

### Script Design

- Keep example scripts runnable as direct entry points with `python script_name.py`.
- Prefer `argparse` for user-facing parameters.
- Keep default argument values consistent with the adjacent README examples.
- Avoid introducing hidden environment assumptions. If a script needs login, hardware, model
  downloads, or external packages, document that explicitly in the README.
- Keep preprocessing, postprocessing, and model-loading logic close to the example unless there is
  already a local helper module such as `imagenet.py`, `coco.py`, `utils.py`, or `visualize.py`.

### Dependency Awareness

- This repo depends on external SDK components such as `qbcompiler`, `qbruntime`, Mobilint NPU
  drivers, Docker images, and gated datasets that may not be available in the local environment.
- Check nearby `requirements.txt` files before adding imports to LLM, STT, or VLM examples.
- Do not assume heavy packages are available globally; document new dependencies close to the
  tutorial that needs them.

## Validation Guide

Use the smallest validation that meaningfully covers the change.

### Documentation-Only Changes

- Check formatting visually.
- Verify referenced paths and filenames exist.
- Re-read commands to ensure flags and defaults match the scripts.

### Python Changes

Run targeted checks on touched files when dependencies allow:

```bash
ruff check path/to/file.py
ruff format path/to/file.py
python -m py_compile path/to/file.py
```

If a change spans several simple scripts in one area, `python -m compileall <directory>` is also
reasonable.

### Hardware-Dependent Work

- Full end-to-end validation may require a Mobilint NPU, installed SDK wheels, Docker images,
  Hugging Face authentication, or large model artifacts.
- Do not claim runtime or compilation execution unless you actually ran it.
- If hardware or proprietary dependencies are unavailable, stop at static validation and state the
  limitation clearly.

## Safety Rules

- Check `git status --short` before editing because this repo may already contain user work.
- Do not revert unrelated changes in the working tree.
- Keep edits focused on the tutorial area the user asked about.
- Avoid broad repo-wide rewrites of Markdown style unless explicitly requested.
- Prefer fixing local inconsistencies over introducing new repo-wide conventions that the surrounding
  docs do not follow.

## Commit Guidance

- Use Conventional Commit prefixes such as `docs:`, `fix:`, `feat:`, `refactor:`, or `chore:`.
- Keep commits scoped to one tutorial area or one documentation pass when possible.
- Mention the affected tutorial or model family in the subject when helpful.
