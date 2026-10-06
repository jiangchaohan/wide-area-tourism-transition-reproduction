from pathlib import Path
import json
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib import colors
ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'generated_figures'
DEST.mkdir(exist_ok=True)
def load(name):return json.loads((ROOT/'results/strengthening/statistical'/f'{name}.json').read_text())
def chart(path,title,categories,series,ymax=1,errors=None):
    c=canvas.Canvas(str(path),pagesize=(620,300));c.setTitle(title);c.setAuthor('')
    x0,y0,w,h=55,67,550,193
    palette=['#6F8497','#286B9E','#159C8C','#BB6B3E','#775799']
    c.setFont('Helvetica-Bold',12);c.drawString(x0,280,title)
    for y in np.linspace(0,ymax,5):
        py=y0+h*y/ymax;c.setStrokeColor(colors.HexColor('#D8DFE5'));c.line(x0,py,x0+w,py)
        c.setFillColor(colors.black);c.setFont('Helvetica',8);c.drawRightString(x0-7,py-3,f'{y:.2f}')
    cell=w/len(categories);bw=cell*.7/len(series)
    for s,(label,values) in enumerate(series.items()):
        for i,value in enumerate(values):
            x=x0+cell*i+cell*.15+bw*s;c.setFillColor(colors.HexColor(palette[s%len(palette)]))
            c.rect(x,y0,bw*.9,h*value/ymax,fill=1,stroke=0)
            if errors and label in errors:
                err=errors[label][i];p=y0+h*value/ymax;c.setStrokeColor(colors.black)
                c.line(x+bw*.45,p-h*err/ymax,x+bw*.45,p+h*err/ymax)
                c.line(x+bw*.2,p+h*err/ymax,x+bw*.7,p+h*err/ymax)
        c.setFillColor(colors.HexColor(palette[s%len(palette)]));c.rect(x0+s*110,15,8,8,fill=1,stroke=0)
        c.setFillColor(colors.black);c.setFont('Helvetica',8);c.drawString(x0+s*110+12,16,label)
    c.setFillColor(colors.black);c.setFont('Helvetica',9)
    for i,label in enumerate(categories):
        lines=label.split('\n')
        for j,line in enumerate(lines):c.drawCentredString(x0+cell*(i+.5),y0-15-j*11,line)
    c.save()

def figures():
    main=load('main');diag=json.loads((ROOT/'results/strengthening/neural_diagnostics.json').read_text())
    m={**main['metrics'],**diag['metrics']}
    chart(DEST/'Figure_3.pdf','Chronological test accuracy', ['Hit@10','NDCG@10'],
          {k:[m[k][v] for v in ['Hit@10','NDCG@10']] for k in ['F1','VOM','TD-VOM','GRU','Transformer']},
          errors={k:[m[k][v+'_std'] for v in ['Hit@10','NDCG@10']] for k in ['GRU','Transformer']})
    trials=main['tuning']['TD-VOM']['trials']
    chart(DEST/'Figure_4.pdf','TD-VOM support sensitivity', [str(a['support']) for a in trials],
          {'Validation':[a['validation_NDCG@10'] for a in trials],'Test':[a['test_NDCG@10'] for a in trials]},ymax=.7)
    groups=['template_unseen','context_unseen','edge_unseen']
    series={k:[main['subgroups'][g]['models'][k]['Hit@10'] for g in groups] for k in ['F1','VOM','TD-VOM']}
    series.update({k:[diag['subgroups'][k][g]['Hit@10'] for g in groups] for k in ['GRU','Transformer']})
    chart(DEST/'Figure_5.pdf','Prediction boundaries under fragment novelty',
          ['Unseen template\n104,919 events','Unseen context\n2,605 events','Unseen connection\n2,699 events'],series)

if __name__=='__main__':figures()
