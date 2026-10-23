# memcpy_async
Use asynchronous global-to-LDS copy to overlap movement with processing of a prior tile. Accelerated MI400-class behavior can avoid the same register-mediated path used by fallback implementations; always verify target behavior and correctness.
