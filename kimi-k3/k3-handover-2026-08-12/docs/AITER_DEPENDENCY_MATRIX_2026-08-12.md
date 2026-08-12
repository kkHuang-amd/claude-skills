# Kimi-K3 AITER dependency matrix — 2026-08-12

```text
Capability              AITER source       Fresh AITER commit  SGLang commit  Default
fused KDA + f_b         #4495              89fce019           base 8950e2e  OFF
MoE caller output       #4617              44eaa76d           61c39c7      OFF via K3 opt
stage1 scratch reuse    #4647              d5411148/2845f200  none          OFF
M16384 BF16 profile     local              d599525e           none          OFF
MLA gate                #4497              11460ce4           1544286      OFF
KDA group64             #4499              b89954d8           9aeff1f      OFF
FP8 preroute/shared     #4504              98ce357d           e13b8a7      OFF
FP8 latent tail         #4503              076b2720           fa81245      OFF
B2 extensions           local/#4499/#4504  2eea7204           81ce307      OFF
A4W4 C16 profile        #4603 closed        61bf56be branch    none          OFF
```

Ownership:

```text
AITER: kernels, public op APIs, scratch lifecycle, tuned configs
SGLang: adapters, flags, packing, warmup, model dispatch and fallback
```

No migration commits were pushed.

## SGLang-owned follow-up

Kimi-specific kernel rows now have an alternative SGLang implementation:

```text
fused KDA + f_b    SGLang 13e6937
MLA gate           SGLang 13e6937
group64            SGLang 13e6937
preroute/shared    SGLang 13e6937
latent tail        SGLang 13e6937
B2 extensions      SGLang 13e6937
M16384 profile     SGLang 433af0d
```

Only these remain hard AITER dependencies:

```text
#4617 fused_moe caller output
#4647 stage1 scratch lifecycle
AITER FlyDSL helper/toolchain modules
```

Details:
[`SGLANG_VENDOR_FLYDSL_2026-08-12.md`](SGLANG_VENDOR_FLYDSL_2026-08-12.md).
