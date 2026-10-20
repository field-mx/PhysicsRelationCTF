"""V3 topology only; unchanged v2 model, optimizer, data, loss and test lock."""
import train_v2 as engine
from public_physics_v3 import build_graph
from data import ROOT

original_hashes=engine.source_hashes
def source_hashes():
    hashes=original_hashes()
    for name in ['public_physics_v3.py','train_v3.py']:
        path=ROOT/name;hashes[str(path)]=engine.sha(path)
    return hashes

engine.build_graph=build_graph
engine.source_hashes=source_hashes
if __name__=='__main__':engine.main()
