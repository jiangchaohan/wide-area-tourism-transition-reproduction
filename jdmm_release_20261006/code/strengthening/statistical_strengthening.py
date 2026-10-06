"""Additional, validation-selected experiments on unchanged historical records."""
from pathlib import Path
from collections import Counter, defaultdict
import json, sys, time
import numpy as np

W=Path(__file__).resolve().parents[2]/'results/strengthening'
OUT=W/'statistical';OUT.mkdir(exist_ok=True)
DATA=json.loads((W/'prepared_data.json').read_text(encoding='utf-8'))
N=len(DATA['states']);IDS=np.arange(N)
SUPPORTS=[1,2,3,5,10,20,50,100]
LAMBDAS=[1,5,10,20,50,100]

def events(rows):
    return np.asarray([(ri,s[t-1] if t else -1,s[t],s[t+1])
                       for ri,r in enumerate(rows) for s in [r['ids']] for t in range(len(s)-1)],dtype=np.int32)

def counts(rows,weighted=False):
    first=np.zeros((N,N));second=defaultdict(Counter)
    freq=Counter(r['template_id'] for r in rows)
    for r in rows:
        s=r['ids'];weight=1/freq[r['template_id']] if weighted else 1
        for t in range(len(s)-1):
            first[s[t],s[t+1]]+=weight
            if t:second[(s[t-1],s[t])][s[t+1]]+=weight
    return first,second

def normalize(a):
    d=a.sum(1,keepdims=True)
    return np.divide(a,d,out=np.zeros_like(a),where=d>0)

def ranking(score,current,pop):
    z=score.copy();z[current]=-np.inf
    return np.lexsort((IDS,-pop,-z))[:10]

def static(a,pop):return np.asarray([ranking(row,i,pop) for i,row in enumerate(a)],dtype=np.int16)

def context_cache(second,first,pop,threshold=1,lam=None):
    base=static(first,pop);cache={}
    for ctx,c in second.items():
        support=sum(c.values())
        if lam is None and support<threshold:continue
        z=np.zeros(N)
        for dest,value in c.items():z[dest]=value/support
        if lam is not None:
            z=z*support/(support+lam)+first[ctx[1]]*lam/(support+lam)
            cache[ctx]=ranking(z,ctx[1],pop)
        else:
            positive=[int(i) for i in np.lexsort((IDS,-pop,-z)) if z[i]>0 and i!=ctx[1]]
            for i in base[ctx[1]]:
                if int(i) not in positive:positive.append(int(i))
                if len(positive)>=10:break
            cache[ctx]=np.asarray(positive[:10],dtype=np.int16)
    return base,cache

def predict(ev,base,cache=None):
    if cache is None:return base[ev[:,2]]
    return np.asarray([cache.get((int(p),int(c)),base[c]) for p,c in ev[:,1:3]],dtype=np.int16)

def scores(ev,recs):
    match=recs==ev[:,3,None];hit=match.any(1)
    rank=np.where(hit,match.argmax(1)+1,np.inf)
    return {'hit5':(rank<=5).astype(float),'hit10':hit.astype(float),
            'mrr':np.where(hit,1/rank,0),'ndcg':np.where(hit,1/np.log2(rank+1),0)}

def mean_or_none(x):return float(np.mean(x)) if len(x) else None

def summarize(ev,recs,rows,pop,dist=None):
    z=scores(ev,recs);nr=len(rows)
    route_n=np.bincount(ev[:,0],minlength=nr)
    route_ndcg=np.bincount(ev[:,0],weights=z['ndcg'],minlength=nr)/np.maximum(route_n,1)
    template=defaultdict(list)
    for r,v in zip(rows,route_ndcg):template[r['template_id']].append(v)
    tail=pop<=np.quantile(pop[pop>0],.8)
    return {'events':len(ev),'routes':nr,'Hit@5':mean_or_none(z['hit5']),
            'Hit@10':mean_or_none(z['hit10']),'MRR@10':mean_or_none(z['mrr']),
            'NDCG@10':mean_or_none(z['ndcg']),'macro_route_NDCG@10':mean_or_none(route_ndcg),
            'macro_template_NDCG@10':mean_or_none([np.mean(v) for v in template.values()]),
            'catalog_coverage@10':len(np.unique(recs))/N,
            'long_tail_events':int(tail[ev[:,3]].sum()),
            'long_tail_Hit@10':mean_or_none(z['hit10'][tail[ev[:,3]]]),
            **({'median_recommendation_distance_km':float(np.median(dist[ev[:,2,None],recs]))} if dist is not None else {})}

def cluster_ci(ev,delta,rows,template=False,reps=2000):
    labels=[r['template_id'] if template else i for i,r in enumerate(rows)]
    _,ix=np.unique(labels,return_inverse=True)
    group=ix[ev[:,0]];sums=np.bincount(group,weights=delta);sizes=np.bincount(group)
    rng=np.random.default_rng(2026);est=[]
    for _ in range(reps):
        draw=rng.integers(0,len(sums),len(sums));est.append(sums[draw].sum()/sizes[draw].sum())
    return {'event_weighted_difference':float(delta.mean()),'CI95':np.quantile(est,[.025,.975]).tolist(),
            'cluster_count':len(sums),'bootstrap_repetitions':reps,'cluster':'template' if template else 'route'}

def fit_evaluate(train,val,test,label,extra=True):
    start=time.time();v=events(val);e=events(test)
    a,c=counts(train);wa,wc=counts(train,True);pop=a.sum(0)
    a=normalize(a);wa=normalize(wa);prediction={};tuning={}
    prediction['F1']=predict(e,static(a,pop));prediction['W1']=predict(e,static(wa,pop))
    for name,first,second in [('VOM',a,c),('TD-VOM',wa,wc)]:
        trials=[];best=-1;choice=None
        for threshold in SUPPORTS:
            base,cache=context_cache(second,first,pop,threshold)
            value=float(scores(v,predict(v,base,cache))['ndcg'].mean())
            trials.append({'support':threshold,'validation_NDCG@10':value})
            trials[-1]['test_NDCG@10']=float(scores(e,predict(e,base,cache))['ndcg'].mean())
            trials[-1]['contexts_used']=len(cache)
            if value>best:best=value;choice=(threshold,base,cache)
        threshold,base,cache=choice
        prediction[name]=predict(e,base,cache);tuning[name]={'selected_support':threshold,'trials':trials}
    if extra:
        best=-1;trials=[]
        for lam in LAMBDAS:
            base,cache=context_cache(c,a,pop,lam=lam)
            value=float(scores(v,predict(v,base,cache))['ndcg'].mean());trials.append({'lambda':lam,'validation_NDCG@10':value})
            if value>best:best=value;choice=(lam,base,cache)
        lam,base,cache=choice;prediction['Interpolated-VOM']=predict(e,base,cache)
        tuning['Interpolated-VOM']={'selected_lambda':lam,'trials':trials}
        raw,_=counts(train);u,s,vt=np.linalg.svd(np.log1p(raw),full_matrices=False)
        trials=[];best=-1
        for rank in [16,32,64,128]:
            base=static((u[:,:rank]*s[:rank])@vt[:rank],pop)
            value=float(scores(v,predict(v,base))['ndcg'].mean());trials.append({'rank':rank,'validation_NDCG@10':value})
            if value>best:best=value;choice=(rank,base)
        rank,base=choice;prediction['Log-count-SVD']=predict(e,base)
        tuning['Log-count-SVD']={'selected_rank':rank,'trials':trials}
    metrics={name:summarize(e,r,test,pop) for name,r in prediction.items()}
    result={'split':{name:{'routes':len(rows),'events':sum(len(r['ids'])-1 for r in rows),
                          'first_date':min(r['date'] for r in rows),'last_date':max(r['date'] for r in rows)}
                     for name,rows in [('train',train),('validation',val),('test',test)]},
            'tuning':tuning,'metrics':metrics,'elapsed_seconds':time.time()-start,
            'training_successor_frequency':{'observed_states':int((pop>0).sum()),'zero_frequency_states':int((pop==0).sum()),
                'observed_atmost5':int(((pop>0)&(pop<=5)).sum()),'observed_atmost10':int(((pop>0)&(pop<=10)).sum()),
                'atmost10_including_zero':int((pop<=10).sum()),'long_tail_positive80percentile':float(np.quantile(pop[pop>0],.8))}}
    (OUT/f'{label}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(label,json.dumps({k:round(v['NDCG@10'],6) for k,v in metrics.items()}),flush=True)
    return result,e,prediction,pop,c,raw if extra else None

def distance_matrix():
    lon=np.radians([r['longitude_wgs84'] for r in DATA['states']]);lat=np.radians([r['latitude_wgs84'] for r in DATA['states']])
    z=np.sin((lat[:,None]-lat[None])/2)**2+np.cos(lat[:,None])*np.cos(lat[None])*np.sin((lon[:,None]-lon[None])/2)**2
    return 6371.0088*2*np.arcsin(np.sqrt(np.clip(z,0,1)))

def main():
    rows=DATA['routes'];cut1=int(.7*len(rows));cut2=int(.85*len(rows))
    train,val,test=rows[:cut1],rows[cut1:cut2],rows[cut2:]
    result,e,pred,pop,contexts,raw=fit_evaluate(train,val,test,'main')
    dist=distance_matrix()
    pred['Popularity']=np.asarray([ranking(pop,int(c),pop) for c in e[:,2]],dtype=np.int16)
    pred['Nearest-distance']=predict(e,static(-dist,pop))
    ndcg={k:scores(e,v)['ndcg'] for k,v in pred.items()}
    result['paired_NDCG_intervals']={}
    for name in ['F1','VOM','Interpolated-VOM','Log-count-SVD']:
        result['paired_NDCG_intervals']['TD-VOM minus '+name]=[
            cluster_ci(e,ndcg['TD-VOM']-ndcg[name],test,template=x) for x in [False,True]]
    train_templates={r['template_id'] for r in train}
    order2=e[:,1]>=0
    seen_context=np.asarray([(int(p),int(c)) in contexts for p,c in e[:,1:3]])
    seen_edge=raw[e[:,2],e[:,3]]>0
    train_triplets={(p,c,d) for (p,c),targets in contexts.items() for d in targets}
    triplets=[tuple(map(int,x)) for x in e[order2,1:4]]
    edge_set=set(map(tuple,e[:,2:4].tolist()));context_set=set(map(tuple,e[order2,1:3].tolist()))
    result['overlap']={'route_start_events':int((~order2).sum()),'order_two_events':int(order2.sum()),
        'unique_test_edges':len(edge_set),'unique_test_contexts_order_two':len(context_set),
        'unique_test_triplets_order_two':len(set(triplets)),
        'edge_event_seen_fraction':float(seen_edge.mean()),
        'unique_edge_seen_fraction':sum(raw[o,d]>0 for o,d in edge_set)/len(edge_set),
        'context_event_seen_fraction_order_two':float(seen_context[order2].mean()),
        'unique_context_seen_fraction_order_two':sum(x in contexts for x in context_set)/len(context_set),
        'triplet_event_seen_fraction_order_two':float(np.mean([x in train_triplets for x in triplets])),
        'unique_triplet_seen_fraction_order_two':sum(x in train_triplets for x in set(triplets))/len(set(triplets))}
    length=np.asarray([len(test[i]['ids']) for i in e[:,0]])
    actual_distance=dist[e[:,2],e[:,3]]
    masks={'template_seen':np.asarray([test[i]['template_id'] in train_templates for i in e[:,0]]),
        'template_unseen':np.asarray([test[i]['template_id'] not in train_templates for i in e[:,0]]),
        'route_start':~order2,'context_seen':order2&seen_context,'context_unseen':order2&~seen_context,
        'edge_seen':seen_edge,'edge_unseen':~seen_edge,
        'target_training_zero':pop[e[:,3]]==0,'target_training_1_20':(pop[e[:,3]]>0)&(pop[e[:,3]]<=20),
        'target_training_over20':pop[e[:,3]]>20,
        'length_2_10':length<=10,'length_11_30':(length>10)&(length<=30),'length_over30':length>30,
        'distance_atmost50km':actual_distance<=50,'distance_50_200km':(actual_distance>50)&(actual_distance<=200),
        'distance_over200km':actual_distance>200}
    result['subgroups']={}
    for group,mask in masks.items():
        result['subgroups'][group]={'events':int(mask.sum()),'routes':len(np.unique(e[mask,0])),
            'models':{name:{'NDCG@10':mean_or_none(ndcg[name][mask]),
                           'Hit@10':mean_or_none(scores(e,r)['hit10'][mask])} for name,r in pred.items()}}
    result['metrics']={name:summarize(e,r,test,pop,dist) for name,r in pred.items()}
    template_unseen=masks['template_unseen']
    result['template_unseen_metrics']={name:{'Hit@5':mean_or_none(scores(e,r)['hit5'][template_unseen]),
        'Hit@10':mean_or_none(scores(e,r)['hit10'][template_unseen]),
        'MRR@10':mean_or_none(scores(e,r)['mrr'][template_unseen]),
        'NDCG@10':mean_or_none(ndcg[name][template_unseen])} for name,r in pred.items()}
    support=np.asarray([sum(contexts.get((int(p),int(c)),{}).values()) for p,c in e[:,1:3]])
    result['support_bins']={}
    for label,mask in [('unseen',support==0),('1--4',(support>=1)&(support<5)),
                       ('5--19',(support>=5)&(support<20)),('20--99',(support>=20)&(support<100)),
                       ('100+',support>=100)]:
        mask=mask&order2
        result['support_bins'][label]={'events':int(mask.sum()),'VOM_minus_F1_NDCG@10':mean_or_none((ndcg['VOM']-ndcg['F1'])[mask])}
    # Archive tie conventions are compared on exactly the same split and selected supports.
    sys.path.insert(0,str(W.parents[1]/'code'))
    import run_route_experiments as old
    result['archive_tie_sensitivity']={}
    for name,weighted in [('F1',False),('W1',True),('VOM',False),('TD-VOM',True)]:
        a,c=counts(train,weighted);base=old.topk_rows(normalize(a))
        cache=old.second_order_candidates(c,base,N,result['tuning'][name]['selected_support']) if name in ['VOM','TD-VOM'] else None
        rec=predict(e,base,cache)
        result['archive_tie_sensitivity'][name]={'NDCG@10':float(scores(e,rec)['ndcg'].mean()),
            'new_minus_archive_NDCG@10':float((ndcg[name]-scores(e,rec)['ndcg']).mean())}
    (OUT/'main.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(OUT/'main_predictions.npz',events=e,**{k.replace('-','_'):v for k,v in pred.items()})
    for year in [2019,2020,2021]:
        t=[r for r in rows if r['date']<f'{year}-01-01']
        v=[r for r in rows if f'{year}-01-01'<=r['date']<f'{year}-07-01']
        x=[r for r in rows if f'{year}-07-01'<=r['date']<f'{year+1}-01-01']
        fit_evaluate(t,v,x,f'rolling_{year}',extra=False)
    fit_evaluate(*[[r for r in s if not r['flagged']] for s in [train,val,test]],'exclude_flagged',extra=False)
    fit_evaluate(*[[r for r in s if len(r['ids'])<=20] for s in [train,val,test]],'length_atmost20',extra=False)
    print('STATISTICAL COMPLETE',flush=True)

if __name__=='__main__':main()
