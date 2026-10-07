"""TTT (total tok/s per GPU) and P90 interactivity from an aiperf profile_export_aiperf.json. usage: ttt_p90.py <json> [gpus=4]"""
import json, sys
d = json.load(open(sys.argv[1])); g = int(sys.argv[2]) if len(sys.argv) > 2 else 4
v = lambda k: d[k]["avg"] if isinstance(d[k], dict) else d[k]
dur = v("benchmark_duration"); isl = v("total_isl"); osl = v("total_osl")
print(f"TTT/GPU {(isl + osl) / dur / g:,.1f}  P90 intvty {d['output_token_throughput_per_user']['p10']:.1f}  "
      f"requests {v('request_count'):.0f}  dur {dur:.0f}s  TTFT p50 {d['time_to_first_token']['p50']:.0f}ms")
