"""Fire 64 fixed, diverse prompts concurrently at an OpenAI-compatible server so
they batch into T==64 decode steps. Same prompts -> identical token ids on both
engines, enabling apples-to-apple router comparison."""
import sys, json, time, threading
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
N = 64
MAX_TOKENS = 256
URL = f"http://127.0.0.1:{PORT}/v1/completions"

def _model_id():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=10) as r:
            return json.loads(r.read())["data"][0]["id"]
    except Exception:
        return "x"
MODEL_ID = _model_id()

TOPICS = [
    "the history of the Roman aqueduct system", "how photosynthesis converts sunlight",
    "the rules of championship chess", "the migration of arctic terns",
    "quantum entanglement in simple terms", "the economics of supply and demand",
    "how a jet engine produces thrust", "the plot of a noir detective film",
    "the chemistry of bread fermentation", "tectonic plate boundaries and earthquakes",
    "the design of suspension bridges", "the life cycle of a monarch butterfly",
    "how vaccines train the immune system", "the structure of a symphony orchestra",
    "the invention of the printing press", "deep sea bioluminescent creatures",
    "the mathematics of prime numbers", "how coral reefs form over centuries",
    "the philosophy of stoicism", "the engineering of a Formula 1 car",
    "the formation of hurricanes", "how neurons transmit signals",
    "the art of Japanese tea ceremony", "the geology of the Grand Canyon",
    "the basics of orbital mechanics", "how lithium batteries store energy",
    "the spread of the Silk Road trade", "the anatomy of a honeybee colony",
    "the principles of thermodynamics", "the cultivation of coffee beans",
    "the architecture of Gothic cathedrals", "how glaciers carve valleys",
    "the evolution of flightless birds", "the dynamics of a stock market crash",
    "the craft of violin making", "how the internet routes packets",
    "the biology of deep hibernation", "the legend of the lost city of gold",
    "the physics of a rainbow", "the cultivation of bonsai trees",
    "the mechanics of human memory", "the discovery of penicillin",
    "the migration patterns of whales", "how solar panels generate electricity",
    "the strategy of ancient naval battles", "the chemistry of fireworks colors",
    "the formation of limestone caves", "how birds navigate by stars",
    "the design of medieval castles", "the science of volcanic eruptions",
    "the history of jazz improvisation", "how the heart pumps blood",
    "the engineering of skyscrapers", "the behavior of wolf packs",
    "the principles of aerodynamics", "the making of fine chocolate",
    "the formation of the solar system", "how earthquakes are measured",
    "the art of glassblowing", "the ecology of rainforests",
    "the mathematics of cryptography", "the domestication of horses",
    "the physics of black holes", "the brewing of traditional sake",
]

_TEMPLATES = [
    "Explain in detail {t}, with mechanisms and an example.",
    "Write a short story whose central theme involves {t}.",
    "List the top five misconceptions about {t} and correct each.",
    "Compare and contrast two opposing views on {t}.",
    "Give step-by-step instructions related to {t}.",
    "Summarize the historical development of {t}.",
    "Describe {t} as if teaching a curious ten-year-old.",
    "Analyze the economic and social impact of {t}.",
]
PROMPTS = [_TEMPLATES[i % len(_TEMPLATES)].format(t=TOPICS[i]) for i in range(N)]

results = [None] * N

def fire(i):
    body = json.dumps({
        "model": MODEL_ID, "prompt": PROMPTS[i], "max_tokens": MAX_TOKENS,
        "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            results[i] = r.status
    except Exception as e:
        results[i] = f"err {e}"

t0 = time.time()
threads = [threading.Thread(target=fire, args=(i,)) for i in range(N)]
for t in threads: t.start()
for t in threads: t.join()
ok = sum(1 for r in results if r == 200)
print(f"done {ok}/{N} ok in {time.time()-t0:.1f}s  sample={results[:3]}")
