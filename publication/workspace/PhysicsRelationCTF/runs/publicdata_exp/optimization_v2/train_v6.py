"""V6 changes only the physical temperature representation, not CTF."""
from pathlib import Path
import train_v5
from model_v6 import PublicPhysicsCTF

engine = train_v5.engine
old_hashes = engine.source_hashes
def source_hashes():
    hashes = old_hashes()
    for name in ['model_v6.py', 'train_v6.py']:
        path = Path(__file__).resolve().parent / name
        hashes[str(path)] = engine.sha(path)
    return hashes
engine.PublicPhysicsCTF = PublicPhysicsCTF
engine.source_hashes = source_hashes
if __name__ == '__main__':engine.main()
