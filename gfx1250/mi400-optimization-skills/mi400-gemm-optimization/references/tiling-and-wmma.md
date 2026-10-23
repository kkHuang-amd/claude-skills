# Tiling and WMMA
- N/M tiling trades data reuse against VGPR pressure and serialized work.
- Macro-tiles let waves cooperatively stage reusable A/B regions but consume LDS and synchronization.
- MI400/GFX12 WMMA mapping should be retuned rather than treated as a drop-in replacement for an older MFMA schedule.
- Watch wave-tail effects when dimensions cross tile boundaries.
