import json, glob, os, re

PAT = re.compile(r"isl(\d+)_osl(\d+)_c(\d+)\.jsonl$")
def load(d):
    out = {}
    for p in glob.glob(os.path.join(d, "*isl*_osl*_c*.jsonl")):
        m = PAT.search(os.path.basename(p))
        if not m: continue
        isl, osl, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        lines = [l for l in open(p) if l.strip()]
        if not lines: continue
        j = json.loads(lines[-1])
        out[(isl, osl, c)] = dict(
            total=j.get("total_throughput", 0.0), out=j.get("output_throughput", 0.0),
            ttft=j.get("median_ttft_ms", 0.0), tpot=j.get("median_tpot_ms", 0.0),
            itl=j.get("median_itl_ms", 0.0))
    return out

SGL_TP8 = load("/workspace/bench_r08/tp8")
SGL_DP8 = load("/workspace/bench_r08/tp8dp8")

# ATOM Experiment 1 (ratio 0.8, ATOM client, np=conc*8) from EXPERIMENT_LOG.md
# (isl,osl,conc) -> (total, out, ttft, tpot, itl)
ATOM_TP8 = {
 (1024,1024,2):(241,121,207.7,16.23,16.06),(1024,1024,4):(453,228,207.4,16.65,16.14),
 (1024,1024,8):(840,420,209.5,18.01,16.83),(1024,1024,16):(1485,739,208.8,21.18,18.66),
 (1024,1024,32):(2435,1214,209.3,25.69,20.54),(1024,1024,64):(3628,1819,215.1,34.87,25.33),
 (8192,1024,4):(1909,215,332.2,17.99,17.12),(8192,1024,8):(3364,379,332.6,19.78,17.75),
 (8192,1024,16):(5717,634,341.6,23.96,19.51),(8192,1024,32):(9028,1001,367.9,30.39,21.23),
 (8192,1024,64):(12523,1397,376.5,44.94,26.16),
}
ATOM_DP8 = {
 (1024,1024,64):(3702,1856,821.3,32.67,29.55),(1024,1024,128):(6368,3181,629.0,38.15,33.91),
 (1024,1024,256):(11093,5543,499.2,44.02,37.26),(1024,1024,512):(16759,8381,511.9,59.43,44.75),
 (1024,1024,1024):(23158,11583,632.0,83.81,56.28),
 (8192,1024,64):(13044,1455,1671.3,41.77,30.12),(8192,1024,128):(19938,2212,1761.3,54.36,33.92),
 (8192,1024,256):(27809,3085,1710.7,79.37,38.22),(8192,1024,512):(34036,3783,2682.0,128.07,45.31),
}

def wl(isl,osl): return f"{isl//1024}k/{osl//1024}k"

def tbl(title, sgl, atom, concs_by_wl):
    print(f"\n### {title}")
    print(f"{'workload':>8} {'conc':>5} | {'SGL out':>8} {'ATOM out':>8} {'SGL/ATOM':>9} | "
          f"{'SGL ITL':>8} {'ATOM ITL':>8} | {'SGL TTFT':>9} {'ATOM TTFT':>9}")
    print("-"*86)
    for (isl,osl), cs in concs_by_wl:
        for c in cs:
            k=(isl,osl,c)
            if k not in sgl or k not in atom: continue
            s=sgl[k]; a=atom[k]
            ratio=s["out"]/a[1] if a[1] else 0
            print(f"{wl(isl,osl):>8} {c:>5} | {s['out']:>8.0f} {a[1]:>8.0f} {ratio*100:>8.0f}% | "
                  f"{s['itl']:>8.1f} {a[4]:>8.1f} | {s['ttft']:>9.0f} {a[2]:>9.0f}")

tbl("tp8  (SGLang+fix  vs  ATOM)", SGL_TP8, ATOM_TP8,
    [((1024,1024),[2,4,8,16,32,64]), ((8192,1024),[4,8,16,32,64])])
tbl("tp8+dp8  (SGLang+fix  vs  ATOM)", SGL_DP8, ATOM_DP8,
    [((1024,1024),[64,128,256,512,1024]), ((8192,1024),[64,128,256,512])])
