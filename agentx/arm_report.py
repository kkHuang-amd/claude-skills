#!/usr/bin/env python3
"""Compact per-arm report + optional A/B compare for the overnight matrix.

Usage:
  arm_report.py <dir-or-armname> [<baseline-dir-or-armname>]
Prints sanity gates first (errors, dropped, duration, ISL) then headline metrics.
Quote numbers from here or benchmark.log -- never from a monitor notification.
"""
import json, sys, os, glob, re

ROOTS = ['/workspace/results/overnight', '/workspace/results']

def resolve(name):
    if os.path.isdir(name):
        return name
    for r in ROOTS:
        p = os.path.join(r, name)
        if os.path.isdir(p):
            return p
    sys.exit('no such arm dir: %s' % name)

def load(d):
    # exact result json only: dsv4_..._c<N>.json (never any *.json -- power files match)
    c = [f for f in glob.glob(os.path.join(d, 'dsv4_fp4_sglang_*_c*.json'))]
    if not c:
        return None
    return json.load(open(sorted(c)[0]))

def errs(d):
    """errors=, last progress line, and aiperf's OWN coverage verdict.

    aiperf fails a run with exit 1 when profiling metric coverage drops under
    95 % ("Profiling metric coverage below the required 95.0%"). That happens
    when requests are still in flight as the window closes, so their TTFT/ITL
    are never recorded -- which biases the latency percentiles toward the FAST
    requests that did finish. A run that trips it is below aiperf's own
    validity bar and must not be quoted.
    """
    lg = os.path.join(d, 'benchmark.log')
    if not os.path.exists(lg):
        return None, None, None
    e, last, cov = None, None, None
    with open(lg, errors='ignore') as f:
        for ln in f:
            m = re.search(r'errors=(\d+)', ln)
            if m:
                e = int(m.group(1))
            if 'progress |' in ln:
                last = ln.strip()
            if 'metric coverage below the required' in ln:
                # This appears 3x: the ERROR, the fatal restatement, and a boxed
                # summary that word-wraps the percentages onto another line.
                # Keep the first that actually carries numbers, not the last.
                c = re.search(r'TTFT=([\d.]+)%.*?latency=([\d.]+)%', ln)
                if c:
                    cov = cov or c.groups()
                else:
                    cov = cov or ('?', '?')
    return e, last, cov

def stats(d):
    j = load(d)
    if not j:
        return None
    rm = j['request_metrics']
    e, last, cov = errs(d)
    return dict(
        name=os.path.basename(d.rstrip('/')),
        tps=rm['throughput']['per_gpu']['total_tput_tps'],
        dur=rm['throughput']['duration_seconds'],
        isl=rm['tokens']['input']['mean'],
        osl=rm['tokens']['output_actual']['mean'],
        ttft=rm['latency']['ttft']['mean'],
        itl90=rm['latency']['itl']['p90'] * 1000,
        intv90=rm['latency']['intvty']['p90'],
        succ=j['num_requests_successful'],
        total=j['num_requests_total'],
        errdrop=j['request_accounting']['records_error_dropped'],
        cache=(j.get('server_metrics', {}).get('cache') or {}).get('gpu_cache_hit_rate'),
        conc=j.get('conc'),
        errors=e, last=last, cov=cov)

def show(s):
    print('== %s (conc %s)' % (s['name'], s['conc']))
    gates = []
    gates.append(('errors', s['errors'], s['errors'] in (0, None)))
    gates.append(('err_dropped', s['errdrop'], s['errdrop'] == 0))
    gates.append(('duration_s', round(s['dur'], 1), 3500 < s['dur'] < 3750))
    gates.append(('aiperf_coverage', 'FAILED %s/%s' % s['cov'] if s['cov'] else 'ok',
                  s['cov'] is None))
    ok = all(g[2] for g in gates)
    print('   GATES %s  %s' % ('PASS' if ok else '**FAIL**',
          '  '.join('%s=%s%s' % (n, v, '' if k else ' <!>') for n, v, k in gates)))
    print('   tok/s/GPU %10.1f   succ %d/%d   cache %.3f' %
          (s['tps'], s['succ'], s['total'], s['cache'] or -1))
    print('   ISL %9.0f  OSL %7.1f  TTFT %6.2fs  ITLp90 %5.2fms  intvtyp90 %5.2f' %
          (s['isl'], s['osl'], s['ttft'], s['itl90'], s['intv90']))

def cmp(a, b):
    print('== DELTA %s vs %s' % (a['name'], b['name']))
    def pc(x, y):
        return (x - y) / y * 100.0
    rows = [('tok/s/GPU', a['tps'], b['tps']), ('ISL', a['isl'], b['isl']),
            ('OSL', a['osl'], b['osl']), ('succ', a['succ'], b['succ']),
            ('TTFT s', a['ttft'], b['ttft']), ('ITLp90 ms', a['itl90'], b['itl90']),
            ('intvty p90', a['intv90'], b['intv90'])]
    for n, x, y in rows:
        print('   %-11s %10.1f  vs %10.1f   %+7.2f %%' % (n, x, y, pc(x, y)))
    d = abs(pc(a['tps'], b['tps']))
    isl = abs(pc(a['isl'], b['isl']))
    print('   VERDICT: headline delta %.2f %% -> %s' % (d,
          'REAL (>=10 %% bar)' if d >= 10 else
          ('suggestive, under 10 %% bar, above the 5.67 %% replicate spread' if d >= 5.67
           else 'INSIDE NOISE (replicate spread 5.67 %) -- report as null')))
    if isl > 3:
        print('   <!> ISL moved %.1f %% -- different workload, comparison suspect' % isl)
    if a['conc'] != b['conc']:
        print('   <!> DIFFERENT CONCURRENCY -- headline comparison INVALID (conc-and-trace-mix.md 19.6)')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    A = stats(resolve(sys.argv[1]))
    if not A:
        sys.exit('no result json yet in %s' % sys.argv[1])
    show(A)
    if len(sys.argv) > 2:
        B = stats(resolve(sys.argv[2]))
        if not B:
            sys.exit('no result json in baseline %s' % sys.argv[2])
        show(B)
        cmp(A, B)
