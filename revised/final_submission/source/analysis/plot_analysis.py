"""Portrait, source-backed analysis figures; original manuscript figures untouched."""
from common import *
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

for path in (OUT/'source/fonts').glob('*.ttf'): font_manager.fontManager.addfont(str(path))
assert any(f.name=='Times New Roman' for f in font_manager.fontManager.ttflist)
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
                     'mathtext.fontset':'custom','mathtext.rm':'Times New Roman',
                     'mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold'})
COLORS={'Matched':'#426c84','Shuffled':'#a46e78','Learnable':'#679485','Sonographic':'#917c9b','Constant':'#8b8983'}
QUERIES=list(COLORS)
LABELS=['2','3','4A','4B','4C','5']


def save(fig,name,caption):
    fig.text(.08,.025,caption,fontsize=8,va='bottom',wrap=True)
    fig.savefig(OUT/f'figures/{name}.pdf')
    fig.savefig(OUT/f'figures/{name}.png',dpi=150)
    plt.close(fig)


def heat(ax,pivot,title,labels=True):
    im=ax.imshow(pivot.to_numpy(),vmin=0,vmax=1,cmap='Blues',aspect='auto')
    ax.set_xticks(range(6),LABELS);ax.set_xlabel('Recorded BI-RADS category')
    ax.set_yticks(range(len(pivot)),pivot.index if labels else [])
    ax.set_title(title,loc='left')
    for i in range(len(pivot)):
        for j in range(6):
            value=pivot.iloc[i,j]
            ax.text(j,i,f'{value:.3f}',ha='center',va='center',color='white' if value>.65 else '#222222',fontsize=8)
    return im


def class_figures():
    d=pd.read_csv(OUT/'statistics/query_classwise_comparison.csv',dtype={'BI_RADS':str})
    fig,axes=plt.subplots(2,1,figsize=(7.2,8.4));fig.subplots_adjust(left=.19,right=.96,top=.93,bottom=.13,hspace=.45)
    for ax,stage in zip(axes,['A','B']):
        p=d[d.stage==stage].groupby(['query','BI_RADS']).F1.mean().unstack().reindex(index=QUERIES,columns=LABELS)
        heat(ax,p,'Residual '+('ON' if stage=='A' else 'OFF')+' | class-wise F1')
    save(fig,'query_classwise_f1','Mean of three training seeds. Test support: 13, 152, 183, 99, 73, 35 images.\nAll six classes retained; the small BI-RADS 2 class has greater estimation uncertainty.')
    fig,axes=plt.subplots(3,1,figsize=(7.2,10));fig.subplots_adjust(left=.19,right=.96,top=.94,bottom=.11,hspace=.5)
    for ax,metric in zip(axes,['Precision','Recall','F1']):
        p=d[d.stage=='B'].groupby(['query','BI_RADS'])[metric].mean().unstack().reindex(index=QUERIES,columns=LABELS)
        heat(ax,p,'Residual OFF | '+metric)
    save(fig,'query_classwise_metrics','Three-seed means on the same 555 test images. Per-seed scores and support are in the source CSV.\nNo class merging or favorable-seed selection.')


def baseline_figure():
    d=pd.read_csv(OUT/'statistics/final_baseline_comparison.csv')
    means=d[(d.record_type=='mean')&(~d.model.str.contains('uniform|permuted',case=False))].set_index('model')
    sd=d[d.record_type=='sample_SD'].set_index('model')
    order=means.index.tolist()
    fig,axes=plt.subplots(2,2,figsize=(8.1,11));fig.subplots_adjust(left=.24,right=.97,bottom=.12,top=.93,wspace=.58,hspace=.3)
    for ax,metric in zip(axes.flat,METRICS):
        for i,name in enumerate(order):
            color='#426c84' if name.startswith('PGCA') else ('#a46e78' if 'CatBoost' in name else '#8b8983')
            ax.errorbar(means.loc[name,metric],i,xerr=sd.loc[name,metric],fmt='o',color=color,ms=4,capsize=2)
        ax.set_yticks(range(len(order)),order if metric in ['Accuracy','QWK'] else [])
        ax.invert_yaxis();ax.set_title(metric+' ('+('lower' if metric=='MAE' else 'higher')+' is better)',fontsize=10)
        ax.grid(axis='x',alpha=.18)
    save(fig,'baseline_comparison','Three-seed mean +/- sample SD; all major predictive baselines retained, including Static OT as an\nadditional spatial configuration. Historical one-seed permutation/Uniform-QK controls remain separately reported.')


def mechanism_figure():
    d=pd.read_csv(OUT/'statistics/paired_query_comparisons.csv');d=d[(d.seed=='mean_across_seeds')&d.holm_family]
    fig,axes=plt.subplots(4,1,figsize=(7.2,9));fig.subplots_adjust(left=.22,right=.94,bottom=.14,top=.94,hspace=.55)
    for ax,metric in zip(axes,METRICS):
        for i,q in enumerate(['Learnable','Shuffled','Sonographic']):
            r=d[(d.metric==metric)&(d.comparison=='B: Matched - '+q)].iloc[0]
            ax.errorbar(r.difference,i,xerr=[[r.difference-r.ci_low],[r.ci_high-r.difference]],fmt='o',color=COLORS[q],capsize=3)
        ax.axvline(0,color='.5',lw=.7);ax.set_yticks(range(3),['vs Learnable','vs Shuffled','vs Sonographic']);ax.invert_yaxis()
        ax.set_title('Residual OFF | '+metric,loc='left',fontsize=10);ax.set_xlabel('Matched minus control')
    save(fig,'query_source_mechanism','Points: mean paired difference across seeds. Bars: 95% paired risk-group bootstrap CI (10,000 draws).\nAll 12 Holm-adjusted group-swap sensitivity P values exceed 0.05; intervals are unadjusted.')


def probability_figures():
    d=pd.read_csv(OUT/'statistics/query_only_intervention_predictions.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.5,8.5));fig.subplots_adjust(left=.12,right=.96,bottom=.14,top=.92,hspace=.45,wspace=.35)
    for row,stage in enumerate(['A','B']):
        for col,metric in enumerate(['mean_abs_threshold_shift','abs_soft_ordinal_index_shift']):
            ax=axes[row,col]
            for seed,color in zip(SEEDS,['#426c84','#a46e78','#679485']):
                v=np.sort(d[(d.stage==stage)&(d.seed==seed)][metric])
                ax.plot(v,np.arange(1,len(v)+1)/len(v),label=str(seed),color=color)
            ax.set_xscale('symlog',linthresh=1e-7);ax.set_ylabel('Cumulative image fraction');ax.set_xlabel('Absolute probability shift' if col==0 else 'Absolute soft-index shift')
            ax.set_title('Residual '+('ON' if stage=='A' else 'OFF'),loc='left');ax.legend(title='Training seed',fontsize=8)
    save(fig,'query_probability_shift','Historical single-donor map; frozen weights and non-Query paths. Each curve contains 555 images.\nSoft index = 1 + sum of five sigmoid outputs, not a calibrated six-class expected grade.')
    t=pd.read_csv(OUT/'statistics/ordinal_threshold_summary.csv');t=t[t.scope=='historical_single']
    fig,axes=plt.subplots(2,1,figsize=(7.2,7.8));fig.subplots_adjust(left=.14,right=.96,bottom=.14,top=.93,hspace=.4)
    for ax,stage in zip(axes,['A','B']):
        for seed,color in zip(SEEDS,['#426c84','#a46e78','#679485']):
            p=t[(t.stage==stage)&(t.seed==seed)].sort_values('threshold')
            ax.plot(p.threshold,p.mean_absolute_shift,'o-',color=color,label=str(seed))
        ax.set_xticks(range(2,7),['>=2','>=3','>=4','>=5','>=6']);ax.set_xlabel('Threshold on encoded grade (1-6)');ax.set_ylabel('Mean absolute sigmoid shift')
        ax.set_title('Residual '+('ON' if stage=='A' else 'OFF'),loc='left');ax.legend(title='Training seed')
    save(fig,'ordinal_threshold_shift','The head uses five independent binary thresholds. Lines show each seed, not standard errors.\nNo probability repair, softmax or post hoc six-class distribution was introduced.')
    fig,axes=plt.subplots(2,1,figsize=(7.2,7.5));fig.subplots_adjust(left=.15,right=.95,top=.93,bottom=.14,hspace=.4)
    for ax,metric in zip(axes,['hard_changed','abs_soft_ordinal_index_shift']):
        for offset,stage in [(-.12,'A'),(.12,'B')]:
            p=d[d.stage==stage].groupby('seed')[metric].mean().reindex(SEEDS)
            ax.bar(np.arange(3)+offset,p,width=.24,label='Residual '+('ON' if stage=='A' else 'OFF'),color='#426c84' if stage=='A' else '#a46e78')
        ax.set_xticks(range(3),SEEDS);ax.set_xlabel('Training seed');ax.set_ylabel('Hard-label change fraction' if metric=='hard_changed' else 'Mean absolute soft-index shift');ax.legend()
    save(fig,'query_prediction_stability','Historical single-donor Query intervention. All images contribute; zero-change seeds remain visible.\nHard-label stability and continuous probability sensitivity are different outcomes.')


def donor_figure(name,metrics,labels):
    d=pd.read_csv(OUT/'statistics/donor_means_per_seed.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.5,8.5));fig.subplots_adjust(left=.13,right=.96,top=.88,bottom=.14,hspace=.38,wspace=.35)
    for row,stage in enumerate(['A','B']):
        for col,(metric,label) in enumerate(zip(metrics,labels)):
            ax=axes[row,col]
            for seed,color in zip(SEEDS,['#426c84','#a46e78','#679485']):
                p=d[(d.stage==stage)&(d.seed==seed)].sort_values('donor_map')
                ax.plot(range(10),p[metric],'o-',ms=3,lw=.8,color=color,label=str(seed))
            ax.set_xticks(range(10));ax.set_xlabel('Fixed donor map');ax.set_ylabel(label);ax.set_title('Residual '+('ON' if stage=='A' else 'OFF'),loc='left')
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.53,.99),ncol=3,title='Training seed',frameon=False,fontsize=9)
    save(fig,name,'Each point is an image mean from one donor map and one trained checkpoint. The ten maps are\nrepeated interventions, not independent patients or training runs. Cluster intervals are provided in the CSV.')


def donor_plates():
    with PdfPages(OUT/'figures/multi_donor_attention_difference.pdf') as pdf:
        for stage in ['A','B']:
            data=np.load(OUT/f'statistics/intervention_arrays/{stage}_visual_cases.npz')
            positions=data['case_positions'];maps=data['attention'];prob=data['probabilities'];images=data['images'];rois=data['roi']
            chosen=[0,2,3,4]
            large=F.interpolate(torch.from_numpy(maps[chosen].reshape(-1,1,14,14)),(224,224),mode='bilinear',align_corners=False).numpy().reshape(4,555,224,224)
            diff=large[1,positions]-large[0,positions];limit=max(float(np.abs(diff).max()),1e-12)
            fig,axes=plt.subplots(6,5,figsize=(8.27,11.69));fig.subplots_adjust(left=.035,right=.98,top=.92,bottom=.10,hspace=.38,wspace=.10)
            fig.suptitle('Frozen Query intervention | Residual '+('ON' if stage=='A' else 'OFF'),y=.975,fontsize=13)
            fig.text(.5,.947,'Six prespecified cases; seed 42; donor maps 0, 1 and 2',ha='center',fontsize=9)
            for i,j in enumerate(positions):
                roi=rois[i,0]>0;y,x=np.nonzero(roi);bbox=(x.min()-.5,y.min()-.5,x.ptp()+1 if hasattr(x,'ptp') else np.ptp(x)+1,np.ptp(y)+1)
                for col in range(5):
                    ax=axes[i,col]
                    if col<4:
                        ax.imshow(images[i,0],cmap='gray',vmin=0,vmax=1)
                        h=large[col,j];h=(h-h.min())/max(float(np.ptp(h)),1e-12)
                        ax.imshow(h,cmap='magma',vmin=0,vmax=1,alpha=.5)
                        grade=int(1+(prob[chosen[col],j]>=.5).sum())
                        ax.set_xlabel('Case '+str(i+1)+' | pred. '+LABELS[grade-1],fontsize=7)
                    else:
                        difference_image=ax.imshow(diff[i],cmap='RdBu_r',vmin=-limit,vmax=limit)
                        ax.set_xlabel('Donor 0 minus matched',fontsize=7)
                    ax.add_patch(Rectangle(bbox[:2],bbox[2],bbox[3],fill=False,edgecolor='#50d9cc',lw=.7))
                    ax.set_xticks([]);ax.set_yticks([])
                    if i==0:ax.set_title(['Matched','Donor 0','Donor 1','Donor 2','Signed difference'][col],fontsize=9)
            color_axis=fig.add_axes([.67,.079,.28,.009])
            fig.colorbar(difference_image,cax=color_axis,orientation='horizontal',ticks=[-limit,0,limit],format='%.1e')
            color_axis.tick_params(labelsize=6,pad=1)
            fig.text(.04,.081,'Signed difference: upsampled native attention probability',fontsize=7)
            fig.text(.04,.060,'Cyan: manual rectangular ROI. Four-head mean; 14 x 14 maps bilinearly upsampled to 224 x 224.',fontsize=8)
            fig.text(.04,.048,'Overlay colors: per-map min-max intensity. Difference colors: one symmetric mass scale within this page.',fontsize=8)
            fig.text(.04,.031,'All cases were selected before outcomes. Spatial attention changes do not establish causal explanation fidelity.',fontsize=8)
            pdf.savefig(fig);fig.savefig(OUT/f'figures/multi_donor_attention_difference_{stage}.png',dpi=150);plt.close(fig)


def localization_figure():
    d=pd.read_csv(OUT/'statistics/final_localization_comparison.csv')
    fig,ax=plt.subplots(figsize=(7.2,8.4));fig.subplots_adjust(left=.13,right=.96,top=.9,bottom=.19)
    areas=[1,2.5,5,7.5,10,15,20]
    for q in ['Matched','Shuffled','Learnable']:
        m=d[(d['query']==q)&(d.record_type=='mean')].iloc[0];s=d[(d['query']==q)&(d.record_type=='sample_SD')].iloc[0]
        keys=[f'Top_{a:g}pct_IoU' for a in areas];mean=m[keys].to_numpy(float);sd=s[keys].to_numpy(float)
        ax.errorbar(areas,mean,yerr=sd,fmt='o-',capsize=3,color=COLORS[q],label=q)
    ax.axvline(10,color='.65',ls=':',lw=.7);ax.set_xticks(areas);ax.set_xlabel('Selected image area (%)');ax.set_ylabel('Rectangular-ROI IoU');ax.set_title('Residual OFF | equal ROI-KL supervision');ax.legend()
    save(fig,'fixed_area_iou_curve','Mean +/- sample SD across three training seeds; identical 383 images / 290 risk groups.\nTop-10% is the primary area-controlled endpoint. All seven prespecified areas are shown.')


if __name__=='__main__':
    torch.set_num_threads(2)
    class_figures();baseline_figure();mechanism_figure();probability_figures()
    donor_figure('multi_donor_attention_stability',['attention_TV','Spearman'],['Attention total variation','Native-map Spearman correlation'])
    donor_figure('multi_donor_probability_stability',['mean_abs_threshold_shift','hard_changed'],['Mean absolute threshold shift','Hard-label change fraction'])
    donor_plates();localization_figure()
    print('FIGURES_EXPORTED',flush=True)
