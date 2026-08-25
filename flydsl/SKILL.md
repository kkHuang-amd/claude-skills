---
name: flydsl-learning-hub
description: >-
  FlyDSL knowledge hub — layout fundamentals, MLIR/Python insertion points, MFMA
  layouts, kernel design and optimization, profiling, A2A dispatch, MegaMoE and
  DSV4 masked-MoE case studies. Use when writing or debugging FlyDSL kernels,
  reasoning about tensor/MFMA layouts, or looking up a FlyDSL playbook. This is a
  large reference corpus — navigate it, never read it in bulk.
---

# FlyDSL knowledge hub

**Do not read this directory in bulk.** It is 61 markdown files, ~972 KB
(~243k tokens). Navigate to one document.

## Start here

| File | Purpose | Size |
|---|---|---|
| `README.md` | Learning plan, environment, roadmap, how to use the corpus | 8 KB |
| `SOURCE_MAP.md` | Map from topic to source file — use this to locate, not grep | 3 KB |

Read `SOURCE_MAP.md` first when you know the topic; read `README.md` first when
you do not.

## Layout

```text
00-foundations/      00_layout_fundamentals.md
01-playbooks/        01_flydsl_basics.md  01b_mlir_python_insertion_point.md
02-profiling/        02_memory_layout.md
03-a2a-dispatch/     03_mfma_layout.md
04-megamoe/          04_kernel_design_and_opt.md
05-dsv4-masked-moe/  05_exercises.md
06-model-case-studies/
07-ecosystem/
```

## Navigating cheaply

```bash
rg -l '<topic>' flydsl/ | head -10        # which files mention it
rg -n '^#{1,2} ' <file> | cut -c1-80      # that file's section map
sed -n 'A,Bp' <file>                      # just that section
```

Prefer `SOURCE_MAP.md` over a repo-wide `rg` — the map is 3 KB and already
answers "which file covers X".
