"""Read-only frozen inputs and outputs isolated to final_submission."""
import os
import sys
from pathlib import Path
import hashlib
import json

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'query_mechanism_20261102'
BASE = ROOT / 'results_pgca36_pathology_20261013'
SUP = ROOT / 'supplementary_validation_20261014'
ROI = ROOT / 'roi_supervision_20261021'
OUT = Path(__file__).resolve().parents[2]
for name in ['audit', 'statistics', 'figures', 'evidence_archive', 'tables']:
    (OUT / name).mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(STUDY), str(SUP), str(BASE), str(ROI)]
SEEDS = [42, 3407, 2026]
METRICS = ['Accuracy', 'MAE', 'QWK', 'Macro-F1']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + '\n')


def markdown(frame):
    def value(v):
        return f'{v:.6f}' if isinstance(v, float) else str(v)
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---']*len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(value(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)) + '\n'
