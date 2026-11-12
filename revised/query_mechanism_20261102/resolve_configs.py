"""Disambiguate active settings without rewriting checkpoint-embedded provenance."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    protocol=json.loads((HERE/'protocol.json').read_text())
    keys=['cohort','image_size','feature_dim','phrase_dim','image_embed_dim','hidden_dim','attention_heads','dropout',
          'image_global_dropout','normalized_image_cache_dtype','epochs','batch_size','optimizer','learning_rate','weight_decay',
          'gradient_clip_norm','num_workers','split_seed','checkpoint_selection','horizontal_flip_probability',
          'global_fitted_scaler_or_imputer','formal_training_device','scheduler','test_hyperparameter_selection']
    records=[]
    for c in protocol['conditions']:
        run=HERE/'runs'/c['run_id']
        original=json.loads((run/'config.json').read_text())
        done=json.loads((run/'completed.json').read_text())
        reused=done['status']=='SUCCESS_REUSED'
        fitted_commit=json.loads((Path(done['source'])/'config.json').read_text())['code_commit'] if reused else original['training_code_commit']
        resolved={k:original[k] for k in keys}
        resolved.update(c)
        resolved.update(training_code_commit=fitted_commit,controlled_wrapper_commit=original['training_code_commit'],parameter_counts=original['parameter_counts'],
                        active_objective='independent_binary_ordinal_BCE + lambda_ROI * KL(mu||alpha)',
                        active_loss_weights=dict(ordinal=1.,ROI_KL=c['lambda_ROI'],static_OT=0.,SB_path=0.),
                        historical_weights_reused=reused,historical_source=done.get('source'),
                        query_mapping='fixed label-blind split-specific risk-group-consistent donor map' if c['query']=='Shuffled' else 'not applied',
                        query_mapping_used_in_splits=['train','val','test'] if c['query']=='Shuffled' else [],
                        query_mapping_sha256=original['query_mapping_sha256'],
                        pathology_direct_residual='matched case' if c['residual_on'] else 'zero',
                        own_case_pathology_used=c['residual_on'] or c['query']=='Matched',
                        donor_case_pathology_used=c['query']=='Shuffled',
                        ROI_used_in_forward=False,ROI_loss_enabled=c['lambda_ROI']>0,
                        evaluation=dict(bootstrap_replicates=2000,bootstrap_seed=724299,paired_training_seeds=[42,3407,2026],
                                        classification_images=555,classification_risk_groups=380,ROI_images=383,ROI_risk_groups=290,
                                        CI='paired cluster percentile, conditional on fitted models and fixed donor map'),
                        inherited_config_sha256=hashlib.sha256((run/'config.json').read_bytes()).hexdigest(),
                        note='Resolved active settings; original config/checkpoint metadata preserved unchanged. Legacy OT/SB/permutation/bootstrap defaults not read by this study are excluded.')
        assert resolved['active_loss_weights']['static_OT']==resolved['active_loss_weights']['SB_path']==0
        (run/'resolved_config.json').write_text(json.dumps(resolved,indent=2)+'\n')
        records.append(resolved)
    (HERE/'resolved_experiment_configs.json').write_text(json.dumps(records,indent=2)+'\n')
    print('RESOLVED_ACTIVE_CONFIGS',len(records),flush=True)


if __name__=='__main__': main()
