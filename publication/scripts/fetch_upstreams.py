"""Explicit user-operated source retrieval; never overwrites an existing tree."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPOS = {
    'CTF': ('https://github.com/jinkwan1115/ChannelTokenFormer.git',
            'b1c100effb9c27cd25de07e91de7f9005d18d5e7', 'workspace/CTF/ChannelTokenFormer'),
    'PhyGNN': ('https://github.com/DiYi1999/PhyGNN.git',
               '75ab808f4a12da9eab650f17bb38f91d9a571154', 'workspace/PhysicsRelationCTF/runs/physics_ai_comparison_v1/vendor/PhyGNN'),
    'MTGNN': ('https://github.com/nnzhan/MTGNN.git',
              'f811746fa7022ebf336f9ecd2434af5f365ecbf6', 'workspace/PhysicsRelationCTF/runs/physics_ai_comparison_v1/vendor/MTGNN'),
    'PatchTST': ('https://github.com/yuqinie98/PatchTST.git',
                '204c21efe0b39603ad6e2ca640ef5896646ab1a9', 'workspace/PhysicsRelationCTF/runs/physics_ai_comparison_v1/vendor/PatchTST'),
}


def main():
    p = argparse.ArgumentParser(description='Review THIRD_PARTY_NOTICES.md first; upstream terms apply.')
    p.add_argument('--ctf', action='store_true')
    p.add_argument('--comparators', action='store_true')
    a = p.parse_args()
    chosen = (['CTF'] if a.ctf else []) + (['PhyGNN','MTGNN','PatchTST'] if a.comparators else [])
    if not chosen:
        p.error('Select --ctf and/or --comparators explicitly')
    for name in chosen:
        url, commit, rel = REPOS[name]
        dest = ROOT/rel
        if dest.exists():
            raise FileExistsError(f'Refusing to overwrite {dest}; inspect it manually.')
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git','clone','--no-checkout',url,str(dest)], check=True)
        subprocess.run(['git','-C',str(dest),'checkout','--detach',commit], check=True)
        actual = subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'], text=True).strip()
        if actual != commit:
            raise RuntimeError('Pinned revision mismatch')
        print(json.dumps(dict(component=name, commit=commit, destination=rel)))


if __name__ == '__main__':
    main()
