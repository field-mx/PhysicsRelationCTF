"""Isolated V5 entry point; preserve the existing deterministic training engine."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_v4
from model_v5 import PublicPhysicsCTF

engine = train_v4.engine
old_hashes = engine.source_hashes

def source_hashes():
    hashes = old_hashes()
    for name in ['model_v5.py', 'train_v5.py']:
        path = Path(__file__).resolve().parent / name
        hashes[str(path)] = engine.sha(path)
    return hashes

engine.PublicPhysicsCTF = PublicPhysicsCTF
engine.source_hashes = source_hashes
if __name__ == '__main__':
    engine.main()
