# FlyDSL knowledge hub source map

## Consolidation policy

Files outside `flydsl/` were copied, not moved. This preserves:

- existing handover links;
- Cursor skill discovery paths;
- relative links inside the original topic trees;
- old chat prompts that reference absolute source paths.

The original path remains authoritative unless a document in this hub says it
was authored here. Copied documents may need to be refreshed when their source
changes.

## Topic sources

### Foundations

Source:

```text
claude-skills/flydsl/{00_layout_fundamentals,01_flydsl_basics,
01b_mlir_python_insertion_point,02_memory_layout,03_mfma_layout,
04_kernel_design_and_opt,05_exercises}.md
```

Copies: `00-foundations/`.

### Production playbooks

```text
dsv4/FLYDSL_KERNEL_AUTHORING.md
dsv4/FLYDSL_KERNEL_OPT_PLAYBOOK.md
dsv4/megamoe/FLYDSL_KERNEL_DEBUG_TOOLKIT.md
```

Copies: `01-playbooks/`.

The Kimi-K3 performance-pattern document was authored directly in the hub:

```text
01-playbooks/flydsl_performance_patterns_from_kda.md
```

### Profiling

```text
dsv4/TRACE_PROFILING.md
```

Copy: `02-profiling/trace_profiling.md`.

### A2A and dispatch

Sources:

```text
dsv4/FLYDSL_A2A_FINAL_REPORT_2026-07-30.md
dsv4/A2A_EP_HANDOVER.md
dsv4/A2A_TBO_GAP_HANDOVER.md
dsv4/A2A_TBO_GAP_PROMPT.md
dsv4/MORI_EPV2_VS_FLYDSL_A2A_NEXT_CHAT.md
dsv4/MORI_EPV2_NEW_IMAGE_HANDOVER.md
```

Copies: `03-a2a-dispatch/`.

`FLYDSL_A2A_FINAL_REPORT_2026-07-30.md` is the authoritative conclusion;
the TBO gap files are historical working notes.

### MegaMoE

Source tree:

```text
dsv4/megamoe/*.md
```

Copies: `04-megamoe/`.

### DSV4 masked MoE

```text
dsv4/MORI_EP_DECODE_ROOTCAUSE.md
dsv4/MASKED_MOE_GFX950_CHANGES.md
```

Copies: `05-dsv4-masked-moe/`.

### Kimi-K3 case study

Sources:

```text
kimi-k3/HANDOVER.md
kimi-k3/HANDOVER_2026-08-07.md
kimi-k3/STAGE1_HANDOVER_2026-08-06.md
kimi-k3/STAGE2_HANDOVER_2026-08-06.md
kimi-k3/STAGE2_KDA_HANDOVER_2026-08-06.md
kimi-k3/TP8_ROCM_TRACE_ANALYSIS.md
kimi-k3/AIPERF_SUMMARY.md
```

Copies: `06-model-case-studies/kimi-k3/`.

### gfx1250 case study

Selected sources:

```text
gfx1250/SKILL.md
gfx1250/STATUS.md
gfx1250/EXPERIMENT_LOG.md
gfx1250/CHANGES.md
gfx1250/HANDOVER_kernel_fixes.md
gfx1250/HANDOVER_gfx1250_moe_emul.md
gfx1250/HANDOVER_moe_emul_method.md
gfx1250/gfx1250.md
```

Copies: `06-model-case-studies/gfx1250/`.

### GPT-OSS case study

```text
gpt-oss/KNOWN_ISSUES.md
```

Copy: `06-model-case-studies/gpt-oss/`.

### Ecosystem and version coupling

```text
sglang-prefill-coalescer/PROBLEMS.md
sglang-prefill-coalescer/HANDOVER_blockscale_triton_compile.md
```

Copies: `07-ecosystem/`.

## Path normalization

Copied documents that referenced the stale prefix:

```text
/workspace/claude-skills
```

were normalized inside the hub copies to:

```text
/workspace/claude-skills
```

Original source documents were not modified.
