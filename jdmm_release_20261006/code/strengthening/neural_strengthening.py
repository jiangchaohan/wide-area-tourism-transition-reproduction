"""Validation-selected local neural baselines; no test-driven tuning.

These are transparent GRU/SASRec-style implementations, not official code.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import json, time, random
import numpy as np
import torch
from torch import nn

W=Path(__file__).resolve().parents[2]/'results/strengthening'
OUT=W/'neural';OUT.mkdir(exist_ok=True)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)
if DEVICE.type=='cuda':
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False

class GRU(nn.Module):
    def __init__(self,n):
        super().__init__();self.item=nn.Embedding(n+1,64,padding_idx=0)
        self.gru=nn.GRU(64,64,batch_first=True);self.out=nn.Linear(64,n+1)
    def encode(self,x):return self.gru(self.item(x))[0]

class Transformer(nn.Module):
    def __init__(self,n):
        super().__init__();self.item=nn.Embedding(n+1,64,padding_idx=0);self.pos=nn.Embedding(40,64)
        block=nn.TransformerEncoderLayer(64,2,128,dropout=.2,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(block,2,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(64);self.out=nn.Linear(64,n+1)
        self.register_buffer('causal',torch.triu(torch.ones(40,40,dtype=torch.bool),diagonal=1))
    def encode(self,x):
        z=self.item(x)+self.pos(torch.arange(40,device=x.device))[None]
        return self.norm(self.encoder(z,mask=self.causal,src_key_padding_mask=(x==0)))

def segments(rows):
    xs=[];ys=[]
    for r in rows:
        seq=np.asarray(r['ids'],dtype=np.int64)+1
        for start in range(0,len(seq)-1,30):
            begin=max(0,start-10);end=min(start+30,len(seq)-1)
            x=np.zeros(40,dtype=np.int64);y=np.full(40,-100,dtype=np.int64)
            m=end-begin;x[:m]=seq[begin:end];y[start-begin:m]=seq[start+1:end+1]
            xs.append(x);ys.append(y)
    return torch.as_tensor(np.asarray(xs),device=DEVICE),torch.as_tensor(np.asarray(ys),device=DEVICE)

def prefixes(rows):
    xx=[];lengths=[];meta=[]
    for ri,r in enumerate(rows):
        seq=r['ids']
        for t in range(len(seq)-1):
            p=seq[max(0,t-39):t+1];x=np.zeros(40,dtype=np.int64)
            x[:len(p)]=np.asarray(p)+1;xx.append(x);lengths.append(len(p))
            meta.append((ri,seq[t-1] if t else -1,seq[t],seq[t+1]))
    return torch.as_tensor(np.asarray(xx),device=DEVICE),torch.as_tensor(lengths,device=DEVICE),np.asarray(meta,dtype=np.int32)

def predict(model,x,lengths,meta,return_recs=False):
    model.eval();recs=[]
    with torch.no_grad():
        for start in range(0,len(x),256):
            batch=x[start:start+256];m=lengths[start:start+256]
            h=model.encode(batch);idx=torch.arange(len(batch),device=DEVICE)
            scores=model.out(h[idx,m-1]);scores[:,0]=-torch.inf
            scores[idx,batch[idx,m-1]]=-torch.inf
            recs.append((scores.topk(10,dim=1).indices-1).cpu().numpy())
    recs=np.vstack(recs)
    matches=recs==meta[:,3,None];found=matches.any(axis=1)
    rank=np.where(found,matches.argmax(axis=1)+1,np.inf)
    ndcg=np.where(found,1/np.log2(rank+1),0)
    result={'Hit@5':float(np.mean(rank<=5)),'Hit@10':float(found.mean()),
            'MRR@10':float(np.where(found,1/rank,0).mean()),'NDCG@10':float(ndcg.mean())}
    return (result,recs,ndcg) if return_recs else result

def fit(name,lr,seed,n,xtrain,ytrain,xval,lval,mval,max_epochs=30,patience=5):
    tag=f'{name}_lr{lr}_seed{seed}'
    historyfile=OUT/(tag+'_history.json');checkpoint=OUT/(tag+'.pt');summaryfile=OUT/(tag+'_summary.json')
    if summaryfile.exists():return json.loads(summaryfile.read_text()),checkpoint
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if DEVICE.type=='cuda':torch.cuda.manual_seed_all(seed)
    model=(GRU(n) if name=='GRU' else Transformer(n)).to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-6)
    criterion=nn.CrossEntropyLoss(ignore_index=-100)
    best=-np.inf;bad=0;history=[];started=time.time();best_epoch=None
    for epoch in range(1,max_epochs+1):
        model.train();order=torch.randperm(len(xtrain),device=DEVICE);loss_sum=0;tokens=0
        for start in range(0,len(order),128):
            ix=order[start:start+128];x=xtrain[ix];y=ytrain[ix]
            optimizer.zero_grad(set_to_none=True)
            h=model.encode(x);mask=y!=-100
            logits=model.out(h[mask]);loss=criterion(logits,y[mask])
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step();count=int(mask.sum());loss_sum+=float(loss.detach())*count;tokens+=count
        val=predict(model,xval,lval,mval)
        history.append({'epoch':epoch,'loss':loss_sum/tokens,'validation':val,'elapsed_seconds':time.time()-started})
        historyfile.write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(tag,epoch,json.dumps(val),'seconds',round(time.time()-started),flush=True)
        if val['NDCG@10']>best+1e-6:
            best=val['NDCG@10'];best_epoch=epoch;bad=0
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint)
        else:bad+=1
        if bad>=patience:break
    summary={'model':name,'learning_rate':lr,'seed':seed,'best_epoch':best_epoch,
             'epochs_run':len(history),'best_validation_NDCG@10':best,
             'elapsed_seconds':time.time()-started,'max_epochs':max_epochs,'patience':patience,
             'checkpoint_selection':'validation NDCG@10 only','test_used_for_tuning':False}
    summaryfile.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    del model,optimizer
    if DEVICE.type=='cuda':torch.cuda.empty_cache()
    return summary,checkpoint

def main():
    data=json.loads((W/'prepared_data.json').read_text(encoding='utf-8'))
    rows=data['routes'];n=len(data['states']);a=int(len(rows)*.7);b=int(len(rows)*.85)
    train,val,test=rows[:a],rows[a:b],rows[b:]
    xtrain,ytrain=segments(train);xval,lval,mval=prefixes(val)
    assert int((ytrain!=-100).sum())==sum(len(r['ids'])-1 for r in train)
    print('device',DEVICE,'torch',torch.__version__,'training_segments',len(xtrain),flush=True)
    result={'runtime':{'torch':torch.__version__,'numpy':np.__version__,'device':str(DEVICE)},
            'protocol':{'dimension':64,'max_prefix':40,'training_stride':30,'overlap_context':10,
                        'batch_size':128,'seeds':[2026,2027,2028],'learning_rate_grid':[.001,.0003],
                        'max_epochs':30,'patience':5,'hyperparameter_selection':'validation only',
                        'implementation':'local GRU and causal Transformer; not official GRU4Rec/SASRec implementations'},
            'tuning':[],'final':[]}
    chosen={}
    for name in ['GRU','Transformer']:
        trials=[]
        for lr in [.001,.0003]:
            summary,checkpoint=fit(name,lr,2026,n,xtrain,ytrain,xval,lval,mval)
            trials.append((summary,checkpoint));result['tuning'].append(summary)
        chosen[name]=max(trials,key=lambda x:x[0]['best_validation_NDCG@10'])[0]['learning_rate']
        (OUT/'progress.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    # Test is constructed only after validation chooses both model configurations.
    xtest,ltest,mtest=prefixes(test)
    train_templates={r['template_id'] for r in train}
    unseen=np.asarray([test[int(i)]['template_id'] not in train_templates for i in mtest[:,0]])
    for name in ['GRU','Transformer']:
        for seed in [2026,2027,2028]:
            lr=chosen[name];summary,checkpoint=fit(name,lr,seed,n,xtrain,ytrain,xval,lval,mval)
            model=(GRU(n) if name=='GRU' else Transformer(n)).to(DEVICE)
            model.load_state_dict(torch.load(checkpoint,map_location=DEVICE,weights_only=True))
            metrics,recs,ndcg=predict(model,xtest,ltest,mtest,True)
            ufound=(recs[unseen]==mtest[unseen,3,None]);urank=np.where(ufound.any(1),ufound.argmax(1)+1,np.inf)
            u={'Hit@5':float(np.mean(urank<=5)),'Hit@10':float(np.mean(urank<=10)),
               'MRR@10':float(np.where(np.isfinite(urank),1/urank,0).mean()),
               'NDCG@10':float(np.where(urank<=10,1/np.log2(urank+1),0).mean())}
            result['final'].append({**summary,'test':metrics,'template_unseen_test':u})
            np.savez_compressed(OUT/f'{name}_seed{seed}_predictions.npz',events=mtest,
                                recommendations=recs.astype(np.int16),ndcg=ndcg)
            (OUT/'progress.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print('TEST',name,seed,json.dumps(metrics),flush=True)
            del model
            if DEVICE.type=='cuda':torch.cuda.empty_cache()
    result['aggregate']={}
    for name in ['GRU','Transformer']:
        trials=[r for r in result['final'] if r['model']==name]
        result['aggregate'][name]={}
        for subset in ['test','template_unseen_test']:
            result['aggregate'][name][subset]={k:{'mean':float(np.mean([r[subset][k] for r in trials])),
                        'std':float(np.std([r[subset][k] for r in trials],ddof=1))} for k in ['Hit@5','Hit@10','MRR@10','NDCG@10']}
    (OUT/'neural_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('COMPLETE',json.dumps(result['aggregate']),flush=True)

if __name__=='__main__':main()
