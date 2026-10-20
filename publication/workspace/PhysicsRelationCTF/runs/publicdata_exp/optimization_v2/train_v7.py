from pathlib import Path
import train_v6
from model_v7 import PublicPhysicsCTF
engine=train_v6.engine
old_hashes=engine.source_hashes
def source_hashes():
    hashes=old_hashes()
    for name in ['model_v7.py','train_v7.py']:
        path=Path(__file__).resolve().parent/name
        hashes[str(path)]=engine.sha(path)
    return hashes
engine.PublicPhysicsCTF=PublicPhysicsCTF
engine.source_hashes=source_hashes
if __name__=='__main__':engine.main()
