import json, random, time
from pathlib import Path
from urllib import request

ROOT = Path.cwd()
DATA = ROOT / 'training-2022' / 'training-2022'
VALVES = ('AV','MV','PV','TV')
API = 'https://satysam-26-heart-disease-detection.hf.space/predict'
N = 5

def m(y_true, y_pred):
    tp=sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)
    fn=sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)
    tn=sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==0)
    fp=sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)
    n=len(y_true)
    rec=tp/(tp+fn) if (tp+fn) else 0
    spec=tn/(tn+fp) if (tn+fp) else 0
    bal=0.5*(rec+spec)
    acc=(tp+tn)/n if n else 0
    pred=sum(1 for p in y_pred if p==1)/n if n else 0
    true=sum(1 for t in y_true if t==1)/n if n else 0
    return {'samples':n,'accuracy':round(acc,4),'balanced_accuracy':round(bal,4),'recall_normal':round(rec,4),'specificity_abnormal':round(spec,4),'pred_normal_rate':round(pred,4),'true_normal_rate':round(true,4),'normal_rate_gap':round(pred-true,4)}

def body(fname,data,b):
    return b''.join([f'--{b}\r\n'.encode(),f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'.encode(),b'Content-Type: audio/wav\r\n\r\n',data,f'\r\n--{b}--\r\n'.encode()])

by_group={}
for txt in DATA.glob('*.txt'):
    lines=txt.read_text(encoding='utf-8', errors='ignore').splitlines()
    out=next((ln.split(':',1)[1].strip() for ln in lines if ln.startswith('#Outcome:')), None)
    if out not in {'Normal','Abnormal'}: continue
    y=1 if out=='Normal' else 0
    rid=txt.stem
    for v in VALVES:
        wav=DATA/f'{rid}_{v}.wav'
        if wav.exists(): by_group.setdefault((v,y), []).append(wav)

rng=random.Random(7)
sample=[]
for v in VALVES:
    for y in (0,1):
        arr=by_group.get((v,y), [])
        rng.shuffle(arr)
        sample.extend((w,y,v) for w in arr[:N])

yt=[]; yp=[]; th=[]
for i,(wav,y,v) in enumerate(sample,1):
    b='----WebKitFormBoundary7MA4YWxkTrZu0gW'
    req=request.Request(API, data=body(wav.name,wav.read_bytes(),b), method='POST')
    req.add_header('Content-Type', f'multipart/form-data; boundary={b}')
    for a in range(4):
        try:
            with request.urlopen(req, timeout=120) as r:
                p=json.loads(r.read().decode('utf-8'))
            break
        except Exception:
            if a==3: raise
            time.sleep(1.0*(a+1))
    yt.append(y); yp.append(1 if p.get('label')=='Normal' else 0); th.append(float(p.get('decision_threshold',0.0)))

out={'api':API,'sample_size':len(sample),'threshold_mean':round(sum(th)/len(th),6),'overall':m(yt,yp)}
print(json.dumps(out, indent=2))
(ROOT/'hf_post_restart_bias_check.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print('saved=hf_post_restart_bias_check.json')
