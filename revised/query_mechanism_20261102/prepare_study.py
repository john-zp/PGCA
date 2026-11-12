"""Read-only historical audit and label-blind protocol preparation."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results_pgca36_pathology_20261013'
SUP = ROOT/'supplementary_validation_20261014'
ROI = ROOT/'roi_supervision_20261021'
HERE = Path(__file__).resolve().parent
SEEDS = [42,3407,2026]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    (HERE/'design').mkdir(exist_ok=True)
    frozen=json.loads((BASE/'source_lock.json').read_text())
    hashes={}
    for field in ['dataset_hashes','dependency_hashes','source_data_hashes']:
        for name,digest in frozen[field].items():
            path=BASE/name if field=='dataset_hashes' else Path(name)
            assert path.exists(), str(path)
            actual=sha(path)
            assert actual==digest, str(path)
            hashes[str(path)]=actual
    for folder,names in [(BASE,['train_one.py','evaluation.py','experiment_models.py','spatial_losses.py','common_config.json']),
                         (SUP,['query_model.py','statistics_common.py','localization_area.py']),
                         (ROI,['study_common.py','selection_lock.json'])]:
        for name in names:
            hashes[str(folder/name)]=sha(folder/name)
    for seed in SEEDS:
        p=BASE/'runs/PGCA-0'/f'seed_{seed}'
        for name in ['best_checkpoint.pt','config.json','training_curve.json','test_predictions.csv','test_attention_maps.npy','completed.json']:
            hashes[str(p/name)]=sha(p/name)
    frame=pd.read_csv(BASE/'data/dataset_manifest.csv')
    assert len(frame)==3709 and frame.groupby('split').size().to_dict()=={'test':555,'train':2599,'val':555}
    assert frame.groupby('patient_key').split.nunique().max()==1
    assert frame.groupby('group_id').split.nunique().max()==1
    assert frame.groupby('patient_key').embedding_sha256.nunique().max()==1
    assert frame.array_index.tolist()==list(range(3709))
    embeddings=np.load(BASE/'data/pathology_embeddings.npy',mmap_mode='r')
    features=np.load(BASE/'data/features36.npy',mmap_mode='r')
    y=np.load(BASE/'data/y.npy',mmap_mode='r')
    masks=np.load(BASE/'data/roi_masks.npy',mmap_mode='r')
    assert embeddings.shape==(3709,512) and np.isfinite(embeddings).all()
    assert features.shape==(3709,36) and np.isin(features,[0,1]).all()
    assert np.array_equal(y,frame.y) and np.isin(y,np.arange(1,7)).all()
    train_idx=frame.index[frame.split=='train'].to_numpy()
    mean=embeddings[train_idx].astype(np.float64).mean(0).astype(np.float32)
    np.save(HERE/'design/train_mean_embedding.npy',mean)
    donor_index=np.arange(len(frame),dtype=np.int64)
    map_rows=[]
    map_audit={}
    for split,seed in [('train',2026110201),('val',2026110202),('test',2026110203)]:
        part=frame[frame.split==split]
        groups=sorted(part.group_id.unique())
        rng=np.random.default_rng(seed)
        representatives={}
        own_hashes={}
        for group,items in part.groupby('group_id'):
            candidates=items.drop_duplicates('patient_key')
            selected=min(candidates.index,key=lambda i:hashlib.sha256(f'{seed}|{group}|{frame.loc[i,"patient_key"]}'.encode()).hexdigest())
            representatives[group]=int(selected)
            own_hashes[group]=set(items.embedding_sha256)
        for attempt in range(1000):
            order=rng.permutation(groups)
            mapping=dict(zip(order,np.roll(order,1)))
            if all(frame.loc[representatives[mapping[g]],'embedding_sha256'] not in own_hashes[g] for g in groups):
                break
        else:
            raise RuntimeError('No label-blind group-consistent distinct-vector derangement')
        for i,row in part.iterrows():
            donor=representatives[mapping[row.group_id]]
            donor_index[i]=donor
            map_rows.append(dict(index=i,image_id=row.image_id,patient_key=row.patient_key,group_id=row.group_id,
                                 donor_index=donor,donor_patient_key=frame.loc[donor,'patient_key'],
                                 donor_group_id=frame.loc[donor,'group_id'],split=split,mapping_seed=seed))
        map_audit[split]={'groups':len(groups),'images':len(part),'mapping_seed':seed,'attempts':attempt+1,
                          'unique_donor_groups':len(set(mapping.values()))}
    donors=pd.DataFrame(map_rows).sort_values('index')
    assert (donors.group_id!=donors.donor_group_id).all()
    assert (donors.patient_key!=donors.donor_patient_key).all()
    assert donors.groupby('group_id').donor_index.nunique().max()==1
    assert np.array_equal(frame.split,frame.iloc[donor_index].split.to_numpy())
    assert (frame.embedding_sha256.to_numpy()!=frame.iloc[donor_index].embedding_sha256.to_numpy()).all()
    donors.to_csv(HERE/'design/query_donors.csv',index=False)
    np.save(HERE/'design/query_donors.npy',donor_index)
    old_donors=np.load(SUP/'design/query_donors.npy')
    old=pd.DataFrame({'patient':frame.patient_key,'donor_patient':frame.iloc[old_donors].patient_key.to_numpy()})
    map_audit['historical_patients_with_multiple_donors']=int((old.groupby('patient').donor_patient.nunique()>1).sum())
    test=frame[frame.split=='test']
    valid=test.index[np.asarray(masks[test.index]).reshape(555,-1).any(1)]
    assert len(valid)==383 and frame.loc[valid,'group_id'].nunique()==290
    assert set(frame.loc[valid,'image_id'])==set(pd.read_csv(BASE/'mechanism_validation_20261014/localization_roi_subset.csv').image_id)
    candidates=sorted(valid,key=lambda i:hashlib.sha256(f'20261102-visualization|{frame.loc[i,"image_id"]}'.encode()).hexdigest())
    cases=[]
    seen=set()
    for i in candidates:
        if frame.loc[i,'group_id'] not in seen:
            seen.add(frame.loc[i,'group_id'])
            cases.append({'index':int(i),'image_id':frame.loc[i,'image_id'],'group_id':frame.loc[i,'group_id']})
        if len(cases)==6: break
    write('design/visualization_cases.json',{'rule':'first six distinct risk groups ordered by SHA256 of fixed salt plus image_id; no labels/predictions used','seed_shown':42,'cases':cases})
    text_audit(frame)
    config=json.loads((BASE/'common_config.json').read_text())
    selected=json.loads((ROI/'selection_lock.json').read_text())['selected']['PGCA-ROI-KL']['lambda_ROI']
    assert selected==.01
    conditions=[]
    for stage,residual,queries,weight in [('A',True,['Matched','Shuffled','Learnable','Sonographic','Constant'],0.),
                                          ('B',False,['Matched','Shuffled','Learnable','Sonographic','Constant'],0.),
                                          ('C',False,['Matched','Shuffled','Learnable'],selected)]:
        for query in queries:
            for seed in SEEDS:
                conditions.append({'stage':stage,'query':query,'residual_on':residual,'seed':seed,'lambda_ROI':weight,
                                   'run_id':f'{stage}_{query}_{"ON" if residual else "OFF"}_seed{seed}',
                                   'reuse_candidate':stage=='A' and query=='Matched'})
    protocol={'timestamp_unix':time.time(),'base_config':config,'conditions':conditions,
              'formal_condition_count':39,'new_fit_count_if_reuse_passes':36,'optional_C_residual_ON':False,
              'D_status':'BLOCKED_PENDING_TEXT_AUDIT_NO_GENERATOR_RECONSTRUCTION',
              'mapping':map_audit,'mapping_identical_across_training_seeds':True,
              'constant_query':'train-image-mean 512-D embedding projected through initialized prompt_proj and external q_proj; detached 256-D buffer frozen at MHA input for entire fit; internal MHA projection remains shared and trainable',
              'learnable_query':'global 256-D parameter N(0,0.02^2), through shared external q_proj; initialization fork_rng preserves original shared parameter initialization',
              'sonographic_query':'new 36-to-256 Linear/GELU/LayerNorm input adapter then shared external q_proj; no new modality',
              'unused_parameters':'existing unused pathology projector (and unused external q_proj for Constant/OFF) frozen, not padded with dummy parameters',
              'residual_OFF':'zeros at external 256-D residual interface; no other text-to-classifier route',
              'primary_contrasts':['B Matched-Learnable','B Matched-Sonographic','B Matched-Shuffled','C Matched-Learnable Top-10% IoU'],
              'classification_metrics':['Accuracy','MAE','QWK','Macro-F1'],'bootstrap_replicates':2000,'bootstrap_seed':724299,
              'topk_fractions':[.01,.025,.05,.075,.1,.15,.2],'threshold':.5,
              'test_access':'no new test predictions until all condition definitions, seeds, weights and visualization cases locked; no test-adaptive changes',
              'stage_reports':True,'test_set_previously_reused':True,'automatic_shutdown':False}
    write('protocol.json',protocol)
    write('design/data_audit.json',{'status':'PASS','images':3709,'directories':int(frame.patient_key.nunique()),'groups':int(frame.group_id.nunique()),
                                  'split_counts':frame.groupby('split').size().to_dict(),'group_counts':frame.groupby('split').group_id.nunique().to_dict(),
                                  'valid_test_ROI_n':383,'valid_test_ROI_groups':290,'mapping':map_audit})
    write('historical_readonly_hashes.json',hashes)
    (HERE/'split_manifest.csv').write_bytes((BASE/'split_manifest.csv').read_bytes())
    write('environment_probe.json',{'python':sys.version,'executable':sys.executable,
                                   'base_git_commit':subprocess.check_output(['git','-C',str(BASE),'rev-parse','HEAD'],text=True).strip(),
                                   'supplement_git_commit':subprocess.check_output(['git','-C',str(SUP),'rev-parse','HEAD'],text=True).strip(),
                                   'ROI_git_commit':subprocess.check_output(['git','-C',str(ROI),'rev-parse','HEAD'],text=True).strip()})
    (HERE/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print(json.dumps({'status':'PREPARED','conditions':len(conditions),'mapping':map_audit,'ROI_weight':selected}),flush=True)


def text_audit(frame):
    root=ROOT/'data_remediation_20261013/original_materials_E'
    forms={}
    for name in sorted(frame.source_form.unique()):
        wb=openpyxl.load_workbook(root/name,read_only=True,data_only=True)
        ws=wb['bi_rads_answers']
        forms[name]={i:str(row[1] or '').strip() for i,row in enumerate(ws.iter_rows(values_only=True),1) if i>=3}
        wb.close()
    cases=frame.drop_duplicates('patient_key')
    texts=[forms[row.source_form][int(row.source_excel_row)] for row in cases.itertuples()]
    counts={'retained_cases':len(texts),'nonempty':sum(bool(t) for t in texts),
            'explicit_birads_patterns':sum(bool(re.search(r'BI[\s-]*RADS|BIRADS',t,re.I)) for t in texts),
            'benign_or_malignant_literal':sum(bool(re.search(r'良性|恶性',t)) for t in texts),
            'diagnostic_type_terms':sum(bool(re.search(r'癌|瘤|腺病|增生|炎|囊肿|纤维',t)) for t in texts),
            'report_section_markers':sum(bool(re.search(r'镜下所见|肉眼所见|大体所见|免疫组化|病理诊断\s*[:：]',t)) for t in texts),
            'newline_fields':sum('\n' in t for t in texts),
            'length_median':float(np.median([len(t) for t in texts])),
            'length_min':min(map(len,texts)),'length_max':max(map(len,texts))}
    assert counts['nonempty']==len(texts) and counts['explicit_birads_patterns']==0
    candidates=[]
    for directory in [ROOT/'local_project',ROOT/'github_project',root]:
        for parent,dirs,files in os.walk(directory):
            dirs[:]=[d for d in dirs if d not in ['.git','__pycache__','runs','pretrained_cache','node_modules']]
            for name in files:
                path=Path(parent)/name
                if path.suffix.lower() not in ['.py','.ipynb'] or path.stat().st_size>30_000_000: continue
                try:
                    content=path.read_text(encoding='utf-8-sig')
                    if path.suffix=='.ipynb':
                        book=json.loads(content)
                        content='\n'.join(''.join(c.get('source',[])) for c in book.get('cells',[]) if c.get('cell_type')=='code')
                    if re.search(r'qwen|AutoTokenizer|SentenceTransformer',content,re.I):
                        matches=[{'line':i,'keywords':re.findall(r'Qwen[\w./-]*|AutoTokenizer|SentenceTransformer|last_hidden_state|normalize|pooling',line,re.I)}
                                 for i,line in enumerate(content.splitlines(),1) if re.search(r'qwen|AutoTokenizer|SentenceTransformer|pooling',line,re.I)]
                        candidates.append({'path':str(path),'sha256':sha(path),'matches':matches})
                except (ValueError,UnicodeError): pass
    write('design/text_content_audit.json',{'counts':counts,'generator_candidate_sources':candidates,
                                         'counting_scope':'literal text markers, not pathologist-validated semantic classification; no raw text exported',
                                         'formal_composition_training_enabled':False})


if __name__=='__main__':
    main()
