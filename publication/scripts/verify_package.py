"""Verify publication file hashes using only the Python standard library."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/'manifests/files.json').read_text())
    checked = 0
    for row in manifest['files']:
        p = ROOT/row['path']
        if not p.is_relative_to(ROOT) or p.is_symlink() or not p.is_file():
            raise RuntimeError(f'Invalid or missing exported path: {row["path"]}')
        if p.stat().st_size != row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError(f'Export checksum mismatch: {row["path"]}')
        checked += 1
    print(json.dumps(dict(verified_files=checked, note='The manifest does not hash itself; generated local data/upstreams are not release assets.')))


if __name__ == '__main__':
    main()
