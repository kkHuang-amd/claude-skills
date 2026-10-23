"""Compare two greedy_dump.sh outputs: per request, tokens matching before the first divergence,
and the max |logprob| difference over that common prefix."""
import json, sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
for i, (x, y) in enumerate(zip(a, b)):
    n = 0
    while n < min(len(x["ids"]), len(y["ids"])) and x["ids"][n] == y["ids"][n]:
        n += 1
    dlp = max((abs(p - q) for p, q in zip(x["lp"][:n], y["lp"][:n]) if p is not None and q is not None), default=0.0)
    print(f"req {i}: same prefix {n}/{len(x['ids'])} tokens, max |dlogprob| on prefix {dlp:.4f}")
