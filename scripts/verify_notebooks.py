"""Execute default notebook code cells headlessly, without changing notebooks.

Run from the repository root or a notebook directory. Optional experiments
remain disabled. This checks Python execution, not browser widget rendering.
No PySINDy imports are allowed, including accidental transitive dependencies.
"""
import builtins
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['MPLBACKEND'] = 'Agg'
os.environ['MG_RUN_CAP_FROM_NOTEBOOK'] = '0'
os.environ['MG_RUN_FIXED_PARAMETER_PROOF'] = '0'
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: plt.close('all')

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] == 'pysindy':
        raise AssertionError('active notebooks must not depend on PySINDy')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import


def frozen_hashes():
    paths = list((root/'CAP').glob('**/outputs*/*'))
    paths += list((root/'notebooks/mackey_glass/outputs_zero_correction').glob('*'))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


if __name__ == '__main__':
    before = frozen_hashes()
    for path in sorted((root/'notebooks').rglob('*.ipynb')):
        print(f'Executing {path.relative_to(root)}', flush=True)
        namespace = {'display': lambda *args: None, '__name__': '__main__'}
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                code = ''.join(cell['source'])
                exec(compile(code, f'{path.name}:cell-{index}', 'exec'), namespace)
        print(f'PASS: {path.name}', flush=True)
    if before != frozen_hashes():
        raise AssertionError('default notebook execution changed frozen proof artifacts')
    print('All five default notebook paths passed; frozen artifacts are unchanged.')
