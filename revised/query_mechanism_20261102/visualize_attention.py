"""Fixed-case attention plates; no outcome-driven case selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyBboxPatch

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'results_pgca36_pathology_20261013'
LABELS={1:'2',2:'3',3:'4A',4:'4B',5:'4C',6:'5'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False})


def main(preview=False):
    torch.set_num_threads(2)
    case_spec=json.loads((HERE/'design/visualization_cases.json').read_text())
    cases=case_spec['cases'];assert case_spec['seed_shown']==42
    manifest=pd.read_csv(BASE/'data/dataset_manifest.csv')
    test=manifest[manifest.split=='test'].reset_index(drop=True)
    images=np.load(BASE/'data/images.npy',mmap_mode='r')
    masks=np.load(BASE/'data/roi_masks.npy',mmap_mode='r')
    conditions=json.loads((HERE/'protocol.json').read_text())['conditions']
    stages=['A'] if preview else ['A','B','C']
    output=HERE/('attention_preview_A.pdf' if preview else 'query_attention_visualization.pdf')
    (HERE/'analysis').mkdir(exist_ok=True)
    with PdfPages(output) as pdf:
        for stage in stages:
            runs={q:HERE/'runs'/next(c['run_id'] for c in conditions if c['stage']==stage and c['query']==q and c['seed']==42)
                  for q in ['Matched','Shuffled','Learnable']}
            predictions={q:pd.read_csv(p/'test_predictions.csv').set_index('image_id') for q,p in runs.items()}
            maps={q:F.interpolate(torch.from_numpy(np.load(p/'test_attention_maps.npy')[:,None]),(224,224),mode='bilinear',align_corners=False)[:,0].numpy()
                  for q,p in runs.items()}
            fig,axes=plt.subplots(6,4,figsize=(8.27,11.69))
            fig.subplots_adjust(left=.065,right=.98,bottom=.10,top=.92,wspace=.08,hspace=.32)
            name={'A':'Residual ON; no spatial supervision','B':'Residual OFF; no spatial supervision','C':'Residual OFF; ROI-KL weight 0.01'}[stage]
            fig.suptitle(f'Query-source comparison | {name}',fontsize=11,y=.975)
            fig.text(.5,.947,'Same six prespecified cases; seed 42; selected checkpoints',ha='center',fontsize=8)
            for row,case in enumerate(cases):
                index=case['index'];image_id=case['image_id']
                assert manifest.loc[index,'image_id']==image_id
                j=int(test.index[test.image_id==image_id][0])
                image=np.asarray(images[index,0],dtype=np.float32)
                roi=np.asarray(masks[index,0])>0
                y,x=np.nonzero(roi)
                bbox=(x.min()-.5,y.min()-.5,x.max()-x.min()+1,y.max()-y.min()+1)
                for col,query in enumerate([None,'Matched','Shuffled','Learnable']):
                    ax=axes[row,col]
                    ax.imshow(image,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                    if query:
                        heat=maps[query][j]
                        norm=(heat-heat.min())/max(float(heat.max()-heat.min()),1e-12)
                        ax.imshow(norm,cmap='magma',vmin=0,vmax=1,alpha=.55,interpolation='nearest')
                        record=predictions[query].loc[image_id]
                        assert int(record.y_true)==int(manifest.loc[index,'y'])
                        ax.set_xlabel('Predicted BI-RADS '+LABELS[int(record.y_pred)],fontsize=7,labelpad=3)
                    else:
                        ax.set_xlabel(f'Case {row+1} | recorded '+LABELS[int(manifest.loc[index,'y'])],fontsize=7,labelpad=3)
                    ax.add_patch(Rectangle(bbox[:2],bbox[2],bbox[3],fill=False,edgecolor='#52ddcd',linewidth=.85))
                    ax.set_xticks([]);ax.set_yticks([])
                    ax.set_xlim(-.5,223.5);ax.set_ylim(223.5,-.5)
                    if row==0: ax.set_title(query or 'Image and reference ROI',fontsize=9,pad=6)
                    for spine in ax.spines.values(): spine.set_visible(False)
            fig.text(.065,.065,'Cyan: manual rectangular reference ROI. Heatmap: four-head mean, 14 x 14 to 224 x 224 bilinear upsampling.',fontsize=7)
            fig.text(.065,.05,'Colors use the same per-map min-max normalization; color intensity is not comparable absolute attention mass.',fontsize=7)
            fig.text(.065,.035,'Cases were fixed by image-ID hash before predictions, without selecting on model performance. Spatial overlap is not causal fidelity.',fontsize=7)
            pdf.savefig(fig)
            fig.savefig(HERE/'analysis'/f'attention_plate_{stage}.png',dpi=180)
            plt.close(fig)
    make_graph()
    print(output,flush=True)


def make_graph():
    fig,ax=plt.subplots(figsize=(8.27,11.69));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    fig.subplots_adjust(left=.03,right=.97,bottom=.03,top=.97)
    ax.text(.5,.97,'Audited PGCA Query / residual pathways',ha='center',fontsize=14,weight='bold')
    def box(x,y,w,h,text,color='#eef2f3'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.005',fc=color,ec='#51585d',lw=.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    def arrow(start,end,label=None):
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=.9,color='#343c40'))
        if label: ax.text((start[0]+end[0])/2,(start[1]+end[1])/2,label,fontsize=7,ha='center',bbox=dict(fc='white',ec='none',pad=1))
    box(.05,.84,.25,.07,'Image\n1 x 224 x 224','#e6eff3')
    box(.375,.84,.25,.07,'Case pathology embedding\n512 dimensions','#f1e7ef')
    box(.70,.84,.25,.07,'Sonographic observations\n36 binary fields','#e7efe8')
    box(.05,.72,.25,.07,'Four non-residual CNN stages\n32 / 64 / 128 / 256','#e6eff3')
    box(.375,.72,.25,.07,'Text projection + external Q\n512 to 256 to 256','#f1e7ef')
    box(.70,.72,.25,.07,'Structured encoder\n36 to 256 to 256','#e7efe8')
    for x in [.175,.5,.825]: arrow((x,.84),(x,.79))
    box(.05,.59,.25,.07,'196 image tokens\nexternal K / V projections','#e6eff3')
    box(.375,.59,.25,.07,'Query-source interface: 256-D\nPathology / learned vector\n36-feature adapter / fixed vector','#f1e7ef')
    arrow((.175,.72),(.175,.66));arrow((.5,.72),(.5,.66),'pathology-derived Query')
    box(.20,.45,.35,.08,'Four-head MHA\ninternal Q / K / V projections\nAttention-weighted image values','#e9e8f1')
    arrow((.175,.59),(.28,.53));arrow((.5,.59),(.46,.53))
    box(.66,.46,.29,.07,'Matched pathology residual\nON: own external Q\nOFF: zero vector','#f1e7ef')
    arrow((.625,.755),(.805,.53),'own case only')
    box(.27,.32,.46,.08,'Add matched residual or zero\nLayerNorm / FFN residual / LayerNorm\n256-dimensional conditioned representation','#e9e8f1')
    arrow((.375,.45),(.43,.40));arrow((.805,.46),(.67,.40))
    box(.16,.19,.68,.08,'Fusion: global image (256) + conditioned representation (256)\n+ structured representation (256) = 768\nFusion MLP: 768 to 256 to 256','#edf0ed')
    arrow((.5,.32),(.5,.27))
    ax.plot([.05,.015,.015],[.755,.755,.23],color='#343c40',lw=.9)
    arrow((.015,.23),(.16,.23))
    ax.text(.015,.49,'global image',rotation=90,ha='center',va='center',fontsize=7,bbox=dict(fc='white',ec='none',pad=1))
    ax.plot([.95,.985,.985],[.755,.755,.23],color='#343c40',lw=.9)
    arrow((.985,.23),(.84,.23))
    ax.text(.985,.64,'features',rotation=90,ha='center',va='center',fontsize=7,bbox=dict(fc='white',ec='none',pad=1))
    box(.21,.065,.58,.07,'Five independent binary threshold logits\nEncoded grade = 1 + count(sigmoid(logit) >= 0.5)\nCodes 1-6 map to BI-RADS 2, 3, 4A, 4B, 4C, 5','#edf0ed')
    arrow((.5,.19),(.5,.135))
    ax.text(.03,.015,'ROI-KL: training-only loss on mean-head attention; no ROI input to the prediction network.\nNo other direct pathology-to-classifier path; selector denotes alternatives, not simultaneous branches.',fontsize=8)
    fig.savefig(HERE/'query_residual_computation_graph.pdf')
    fig.savefig(HERE/'analysis/query_residual_computation_graph.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true')
    main(parser.parse_args().preview)
