# TDM Descriptors
Descriptors express base, element size, tensor dimensions, strides and tile dimensions. Bounds and strides serve different purposes. Offset handling must preserve original physical strides. Prefer explicit, reviewable descriptors over opaque bit manipulation when an appropriate higher-level interface exists.
