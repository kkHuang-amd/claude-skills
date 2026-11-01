import json, sys, threading, urllib.request
n = int(sys.argv[1]); out = [None] * n
def go(i):
    body = json.dumps({"text": f"Q: What is {12+i} + {30+i}? Answer with the number and a short sentence.\nA:",
                       "sampling_params": {"max_new_tokens": 40, "temperature": 0}}).encode()
    req = urllib.request.Request("http://127.0.0.1:30000/generate", body, {"Content-Type": "application/json"})
    out[i] = json.loads(urllib.request.urlopen(req, timeout=900).read())["text"]
ts = [threading.Thread(target=go, args=(i,)) for i in range(n)]
[t.start() for t in ts]; [t.join() for t in ts]
for i, o in enumerate(out[: min(n, 4)]):
    print(i, repr(o[:120]))
