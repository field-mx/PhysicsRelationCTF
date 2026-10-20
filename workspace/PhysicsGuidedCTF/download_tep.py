"""Fetch only the two fault-free Rieth TEP files and original simulator documentation."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parent/'data/tep_raw'

def fetch(name):
    url='https://www.kaggle.com/api/v1/datasets/download/averkij/tennessee-eastman-process-simulation-dataset/'+name
    path=ROOT/name
    if not path.exists():
        tmp=path.with_suffix('.download')
        urllib.request.urlretrieve(url,tmp)
        if zipfile.is_zipfile(tmp):
            with zipfile.ZipFile(tmp) as z:
                names=[n for n in z.namelist() if Path(n).name==name]
                if len(names)!=1: raise ValueError(z.namelist())
                path.write_bytes(z.read(names[0]))
        else:tmp.replace(path)
    return {'file':str(path),'url':url,'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        files=list(pool.map(fetch,['TEP_FaultFree_Training.RData','TEP_FaultFree_Testing.RData']))
    for name in ['README.md','teprob.f','LICENSE']:
        path=ROOT/name
        if not path.exists():urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/'+name,path)
    (ROOT/'download_manifest.json').write_text(json.dumps(files,indent=2))
    print(json.dumps(files,indent=2))
