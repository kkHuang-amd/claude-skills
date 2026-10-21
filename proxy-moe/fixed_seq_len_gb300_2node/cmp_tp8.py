import math, re, sys
def load(path, recipe):
    rows, cur = {}, None
    for line in open(path):
        m = re.match(r'#\s+(\d+)/(\d+)\s+(\S+)', line)
        if line.startswith("#"): cur = bool(m) and m.group(3) == recipe; continue
        if cur and line.startswith('| ') and line[2].isdigit():
            c = [x.strip() for x in line.strip().strip('|').split('|')]
            rows[(int(c[0]), int(c[1]), int(c[3]))] = dict(ttt=float(c[5]), ttft=float(c[7]), itl=float(c[8]), p90=float(c[10]))
    return rows
mi = load(sys.argv[1], 'tp8ep1'); gb = load(sys.argv[2], 'tp8ep1')
keys = sorted(set(mi) & set(gb)); print('matched', len(keys))
gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))
def summ(name, ks):
    r = {f: gm([mi[k][f] / gb[k][f] for k in ks]) for f in ('ttt', 'p90', 'itl', 'ttft')}
    print(f"| {name} | {len(ks)} | {r['ttt']:.3f} | {r['p90']:.3f} | {r['itl']:.3f} | {r['ttft']:.3f} |")
for s in ((1024, 1024), (8192, 1024), (70000, 300)):
    summ(f'{s[0]}/{s[1]}', [k for k in keys if k[:2] == s])
summ('all 3 shapes', keys)
for s in ((1024, 1024), (8192, 1024)):
    summ(f'{s[0]}/{s[1]}, CONC <= 512', [k for k in keys if k[:2] == s and k[2] <= 512])
summ('all, CONC <= 512', [k for k in keys if k[2] <= 512])
print('per point: shape conc | MI TTT/GPU GB TTT/GPU ratio | MI ITL GB ITL | MI TTFT GB TTFT')
for k in keys:
    a, b = mi[k], gb[k]
    print(f"{k[0]}/{k[1]} c{k[2]} | {a['ttt']:.0f} {b['ttt']:.0f} {a['ttt']/b['ttt']:.3f} | {a['itl']:.2f} {b['itl']:.2f} | {a['ttft']:.0f} {b['ttft']:.0f}")
if len(sys.argv) > 3:
    mi4 = load(sys.argv[1], 'tp4ep1'); gb4 = load(sys.argv[3], 'tp4ep1')
    print('TP8/TP4 TTT/GPU ratio (ideal 0.5 = equal step time):  MI355X  GB300')
    for k in [(1024,1024,2),(1024,1024,64),(1024,1024,512),(8192,1024,512),(70000,300,2),(70000,300,32)]:
        print(k, f"{mi[k]['ttt']/mi4[k]['ttt']:.3f}  {gb[k]['ttt']/gb4[k]['ttt']:.3f}")
    print('all-point geomean', f"{gm([mi[k]['ttt']/mi4[k]['ttt'] for k in keys]):.3f}  {gm([gb[k]['ttt']/gb4[k]['ttt'] for k in keys]):.3f}")
