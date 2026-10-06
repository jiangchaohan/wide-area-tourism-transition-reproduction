from pathlib import Path
import ast, json, hashlib
import numpy as np
import pandas as pd

W=Path(__file__).resolve().parent
BASE=W.parents[1]
OUT=BASE/'results/strengthening';OUT.mkdir(exist_ok=True)

def flagged(value):
    if pd.isna(value):return False
    s=str(value).strip()
    return s not in {'','[]','0','False','false','None','nan'}

if __name__=='__main__':
    routes=pd.read_excel(BASE/'data/Supplementary_Data_S2_Cleaned_Routes.xlsx',sheet_name='Cleaned_Model_Routes')
    states=pd.read_excel(BASE/'data/Supplementary_Data_S3_States_and_Transitions.xlsx',sheet_name='Attraction_States').sort_values('state_id')
    names=dict(zip(states.attraction_name_standardized.astype(str),states.state_id.astype(int)))
    prepared=[];excluded=[]
    for row in routes.to_dict('records'):
        seq=[]
        for name in ast.literal_eval(str(row['attraction_sequence_cleaned'])):
            if str(name) in names:
                i=int(names[str(name)])
                if not seq or i!=seq[-1]:seq.append(i)
        date=pd.to_datetime(row['departure_date'],errors='coerce')
        if len(seq)>=2 and pd.notna(date) and 2017<=date.year<=2021:
            prepared.append({'date':date.strftime('%Y-%m-%d'),'record_id':int(row['record_id']),
                             'template_id':int(row['route_template_id']),'ids':seq,
                             'flagged':flagged(row.get('entities_flagged_for_review')),
                             'quality':str(row.get('quality_flag',''))})
        else:excluded.append(int(row['record_id']))
    prepared.sort(key=lambda x:(x['date'],x['record_id']))
    audit={'source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (BASE/'data').glob('*.xlsx')},
           'quality_counts':routes.quality_flag.fillna('<missing>').value_counts().to_dict(),
           'flagged_routes':sum(r['flagged'] for r in prepared),'excluded_record_ids':excluded,
           'year_counts':dict(pd.Series([r['date'][:4] for r in prepared]).value_counts().sort_index()),
           'length_quantiles':{str(q):float(np.quantile([len(r['ids']) for r in prepared],q)) for q in [0,.5,.9,.95,.99,1]},
           'review_metadata':{}}
    file=BASE/'data/Supplementary_Data_S3_States_and_Transitions.xlsx'
    book=pd.ExcelFile(file)
    for sheet in book.sheet_names:
        if 'Review' in sheet:
            data=pd.read_excel(file,sheet_name=sheet)
            audit['review_metadata'][sheet]={'rows':len(data),'columns':list(data.columns),
                'sample':data.head(6).fillna('').to_dict('records')}
    state_rows=states[['state_id','attraction_name_standardized','longitude_wgs84','latitude_wgs84']].to_dict('records')
    (OUT/'prepared_data.json').write_text(json.dumps({'states':state_rows,'routes':prepared},ensure_ascii=False),encoding='utf-8')
    (OUT/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=int),encoding='utf-8')
    print('Prepared',len(prepared),'routes; excluded',excluded)
