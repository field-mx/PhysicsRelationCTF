"""Read only the public normal CSV member; zipfile validates its CRC."""
import hashlib
import io
import json
import zipfile
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
URL = 'https://fdddata.lbl.gov/data/Simulated_LBNL_FDD_Data_Sets_Chiller_Plant/LBNL_FDD_Data_Sets_Chiller_Plant.zip'


class HTTPFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        r = self.session.head(url, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers['Content-Length'])
        self.etag = r.headers.get('ETag')
        self.pos = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        if self.pos < 0: raise ValueError('negative seek')
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.pos+n)
        if end <= self.pos: return b''
        r = self.session.get(self.url, headers={'Range': f'bytes={self.pos}-{end-1}'}, timeout=(30,180))
        r.raise_for_status()
        if r.status_code != 206 or len(r.content) != end-self.pos:
            raise RuntimeError('Server did not honor byte range')
        actual = r.headers.get('Content-Range', '')
        if actual != f'bytes {self.pos}-{end-1}/{self.size}':
            raise RuntimeError(f'Unexpected content range {actual}')
        self.pos = end
        return r.content


def main():
    out = ROOT/'data'
    out.mkdir(exist_ok=True)
    dest = out/'ChillerPlant.csv'
    if dest.exists() and (out/'provenance.json').exists():
        meta = json.loads((out/'provenance.json').read_text())
        if hashlib.sha256(dest.read_bytes()).hexdigest() != meta['csv_sha256']:
            raise RuntimeError('Existing raw CSV changed')
        print('Existing verified download', flush=True)
        return
    remote = HTTPFile(URL)
    with zipfile.ZipFile(remote) as z:
        items = [i for i in z.infolist() if Path(i.filename).name == 'ChillerPlant.csv']
        if len(items) != 1: raise RuntimeError([i.filename for i in z.infolist()])
        member = items[0]
        print(json.dumps({'member':member.filename,'compressed_bytes':member.compress_size,
                          'uncompressed_bytes':member.file_size,'archive_bytes':remote.size}), flush=True)
        content = z.read(member)  # verifies CRC32 before writing
        tmp = dest.with_suffix('.partial')
        tmp.write_bytes(content)
        tmp.replace(dest)
        meta = {'url':URL,'doi':'10.25984/1881324','archive_etag':remote.etag,
                'archive_bytes':remote.size,'member':member.filename,'crc32':member.CRC,
                'csv_sha256':hashlib.sha256(content).hexdigest(),'csv_bytes':len(content),
                'kind':'EnergyPlus-Modelica simulation; normal operation only'}
        (out/'provenance.json').write_text(json.dumps(meta,indent=2))
        print(json.dumps(meta), flush=True)


if __name__ == '__main__': main()
