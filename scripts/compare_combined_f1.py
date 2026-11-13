import json
from wsp.metrics import bootstrap_delta
from wsp.data import load
F='runs/20261111-bern-combined-v1-f1-b10-1'; O='runs/20261106-bern-combined-v1-b10-1'
ho=load('data/heldout/bern-eval.jsonl'); pr=load('data/practice/bern-train-300.jsonl')
pg=lambda it:[x['key'].rsplit('/',1)[0] for x in it]
hand=lambda it:['H' if 'Hertwig' in x['key'] else 'S' for x in it]
b={m['model']:m['per_line'] for m in json.load(open('docs/baselines/heldout-bern-eval.json'))['models']}
f=json.load(open(F+'/summary.json')); o=json.load(open(O+'/summary.json'))
c0=[tuple(x) for x in b['kraken-bern-combined-v1']]; fl=[tuple(x) for x in f['heldout']['per_line']]; ol=[tuple(x) for x in o['heldout']['per_line']]
def cer(r): return sum(a for a,_ in r)/sum(c for _,c in r)
def sub(r,h,k): return [x for x,y in zip(r,h) if y==k]
out={}
def rep(name,a,bb,it):
    d=bootstrap_delta(a,bb,clusters=pg(it)); out[name]=d
    print(f"{name}: {cer(a):.4f} -> {cer(bb):.4f}  delta {d['delta_cer']:+.4f} line {[round(x,4) for x in d['ci95']]} page {[round(x,4) for x in d['ci95_cluster']]}")
H=hand(ho)
rep('heldout fable vs C0',c0,fl,ho); rep('heldout opus vs C0',c0,ol,ho); rep('heldout fable vs opus',ol,fl,ho)
for k in 'HS':
    it=[x for x,y in zip(ho,H) if y==k]
    rep(f'heldout {k} fable vs C0',sub(c0,H,k),sub(fl,H,k),it); rep(f'heldout {k} opus vs C0',sub(c0,H,k),sub(ol,H,k),it)
# practice: per-line from logs
def passes(run):
    L=[json.loads(l) for l in open(run+'/log.jsonl')]; return L
import wsp.metrics as M
def pass_lines(run,n):
    rd=[json.loads(l) for l in open(f'{run}/readings/pass_{n:03d}.jsonl')]
    return M.per_line_edits([r['text'] for r in rd],[x['text'] for x in pr])
import os
fl_last=max(int(p[5:8]) for p in os.listdir(F+'/readings') if p.startswith('pass_'))
# final practice state = last kept pass; find from log
def last_kept_pass(run):
    L=passes(run); kept=1; cur=0
    for e in L:
        if e['kind']=='pass': cur+=1 if False else 0
    p=0; lastpass=None; k=1
    for e in L:
        if e['kind']=='pass': p+=1; lastpass=p
        if e['kind']=='decision' and e['kept']: k=lastpass
    return k
from wsp.kraken_subject import KrakenSubject
PH=hand(pr)
a=pass_lines(F,1)
fk=last_kept_pass(F); fb=pass_lines(F,fk)
os_=KrakenSubject(os.path.expanduser('~/Repo/weightspace/runs/20261106-bern-combined-v1-b10-1/final.mlmodel'))
assert os_.digest()==o['final_digest']
ob=M.per_line_edits([p.text for p in os_.forward(pr)],[x['text'] for x in pr])
print('opus final practice CER (rescored)',cer(ob),'log last-kept', o.get('kept_state'))
for nm,bb in [('fable',fb),('opus',ob)]:
    rep(f'practice {nm} vs C0',a,bb,pr)
    for h in 'HS':
        it=[x for x,y in zip(pr,PH) if y==h]; rep(f'practice {h} {nm} vs C0',sub(a,PH,h),sub(bb,PH,h),it)
rep('practice fable vs opus',ob,fb,pr)
json.dump(out,open(F+'/comparison.json','w'),indent=1)
