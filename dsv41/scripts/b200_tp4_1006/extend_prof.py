"""P2: profile P_L + 512 new tokens over a warmed prefix P_L (conc 1). usage: extend_prof.py <L>"""
import random, sys, time, urllib.request
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefix_sweep import URL, ttft, toks, metrics  # noqa: E402

L = int(sys.argv[1])
rng = random.Random(777 + L)
P = toks(rng, L)
ttft(P)
ttft(P + toks(rng, 512))  # one unprofiled warm extend
m0 = metrics()
urllib.request.urlopen(urllib.request.Request(URL + "/start_profile", method="POST"), timeout=120).read()
time.sleep(1)
t = ttft(P + toks(rng, 512))
time.sleep(1)
urllib.request.urlopen(urllib.request.Request(URL + "/stop_profile", method="POST"), timeout=600).read()
m1 = metrics()
print(f"L={L} N=512 ttft_ms={t:.1f} hit={m1['vllm:prefix_cache_hits_total']-m0['vllm:prefix_cache_hits_total']:.0f}"
      f" query={m1['vllm:prefix_cache_queries_total']-m0['vllm:prefix_cache_queries_total']:.0f}")
