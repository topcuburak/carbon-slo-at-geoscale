"""Replay SLA filter variants over the recorded snapshots.

    python3 replay.py <run_dir> [<run_dir> ...]

The timeseries carries no ISL sums, so prefill_isl is approximated as
count x mean ISL (841 for mlperf_llama3_8B). Everything else -- decoding,
prefilling, pending, prefill_duty, kv_tokens, num_gpus -- is the exact field
the scheduler was handed.
"""
import csv, collections, json, statistics as st, sys, os

FIT=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "datacenter", "perf_models", "analytical", "fit_data",
                 "latency_fit_tp1_llama3_8B.json")
L=json.load(open(FIT)); T,P=L["tpot"],L["ttft"]
ISL, OSL, BUD = 841.0, 71.0, 100.0

def step(bs,K):
    bs=max(bs,1.0)
    return (T["c0"]+T["c1"]*bs+T["c2"]*K+T["c3"]*bs*bs
            +T["c4"]*(K/bs)+T["c5"]*bs*K)
def pfill(N,bs):
    if N<=0: return 0.0
    return (P["b0"]+P["b1"]*N+P["b2"]*max(0,N-P["k1"])+P["b3"]*max(0,N-P["k2"])
            +P["b4"]*max(bs,1)+P["b5"]*max(bs,1)*ISL*ISL)

def variants(dec,pre,pend,duty,kv,eng):
    d,f,p = dec/eng, pre/eng, pend/eng
    K, du = kv/eng, min(0.95,max(0.0,duty))
    out={}
    # A: bugunku kod -- batch=decoding, backlog=prefilling, gelen istegin osl'i
    out["A_current"]=(step(d+1,K)+pfill(f*ISL,f)/OSL)/(1-du)
    # B: onerilen -- batch=active+pending, backlog=prefilling+pending
    out["B_active+pend"]=(step(d+f+p+1,K)+pfill((f+p)*ISL,f+p)/OSL)/(1-du)
    # C: B + incumbent amortismani (kalan ~OSL/2)
    out["C_B+incumbent"]=(step(d+f+p+1,K)+pfill((f+p)*ISL,f+p)/(OSL/2))/(1-du)
    return out

for D in sys.argv[1:]:
    rows=list(csv.DictReader(open(os.path.join(D,"logs","fleet_timeseries.csv"))))
    fl=collections.Counter()
    for r in rows: fl[int(r["epoch"])]+=int(r["decoding"])
    peak=max(fl.values()); live={e for e,v in fl.items() if v>=.5*peak}
    per=collections.defaultdict(list); byep=collections.defaultdict(dict)
    for r in rows:
        e=int(r["epoch"])
        if e not in live: continue
        g=max(int(r["num_gpus"]),1)
        v=variants(int(r["decoding"]),int(r["prefilling"]),int(r["pending"]),
                   float(r["prefill_duty"]),int(r.get("kv_tokens") or 0),g)
        per[r["dc_id"]].append((int(r["decoding"])/g,
                                (int(r["prefilling"])+int(r["pending"]))/g,v))
        byep[e][r["dc_id"]]=v
    hot=sorted(per,key=lambda d:-st.mean(x for x,_,_ in per[d]))[:6]
    q=lambda a,pp: sorted(a)[min(len(a)-1,int(len(a)*pp))]
    print(f"\n=== {D}   {len(live)} plato epoch")
    print(f"{'DC':<17}{'dec/bek korel':>14}"
          +"".join(f"{k+' p10/p50/p90':>26}" for k in ("A_current","B_active+pend")))
    for dc in hot:
        dd=[x for x,_,_ in per[dc]]; bb=[y for _,y,_ in per[dc]]
        try: c=st.correlation(dd,bb)
        except Exception: c=float("nan")
        cells=""
        for k in ("A_current","B_active+pend"):
            v=[z[k] for _,_,z in per[dc]]
            cells+=f"{q(v,.1):>8.0f}{q(v,.5):>9.0f}{q(v,.9):>9.0f}"
        print(f"{dc:<17}{c:>14.2f}{cells}")
    print(f"\n{'variant':<16}{'kabul% (sicak siteler)':>24}"
          f"{'epoch basina kabul eden site':>30}")
    for k in ("A_current","B_active+pend","C_B+incumbent"):
        adm=[z[k]<=BUD for dc in hot for _,_,z in per[dc]]
        nsite=[sum(1 for v in byep[e].values() if v[k]<=BUD) for e in sorted(live)]
        print(f"{k:<16}{100*sum(adm)/len(adm):>23.1f}%"
              f"{st.mean(nsite):>20.1f} / 16   (min {min(nsite)})")
