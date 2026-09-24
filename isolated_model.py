"""One-change hypothesis: physics must not rewrite the shared patch encoder.

The physical branch reads the CURRENT native projection, but its extra path
does not differentiate that projection. Native CTF training is not frozen.
All forward values, parameter names/counts and serialization stay identical.
"""
from types import SimpleNamespace
from relation_model import PhysicsRelationCTF

class IsolatedPhysicsCTF(PhysicsRelationCTF):
    def _prepare_messages(self,embedding,args,output):
        projection=embedding.embedding_dict['pl24']
        view=SimpleNamespace(embedding_dict={
            'pl24':SimpleNamespace(weight=projection.weight.detach())})
        return super()._prepare_messages(view,args,output)
