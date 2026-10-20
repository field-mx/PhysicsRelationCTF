"""Known-boundary adapter; all CTF weights, data, optimization and loss unchanged."""
import train_v2 as engine
from public_physics_v4 import build_graph,fit_gains
from public_model_v4 import PublicPhysicsCTF
from data import ROOT

original_hashes=engine.source_hashes
def source_hashes():
    hashes=original_hashes()
    for name in ['public_physics_v3.py','public_physics_v4.py','public_model_v4.py','train_v4.py','data/control_reference.json']:
        path=ROOT/name;hashes[str(path)]=engine.sha(path)
    return hashes

engine.build_graph=build_graph;engine.fit_gains=fit_gains
engine.PublicPhysicsCTF=PublicPhysicsCTF;engine.source_hashes=source_hashes
if __name__=='__main__':engine.main()
