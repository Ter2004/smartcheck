"""Download checksum-verified LFW originals and development test pairs locally.

Sources/checksums: scikit-learn sklearn/datasets/_lfw.py.
Images go to the ignored .build directory by default; no database access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

FILES = {
    'lfw.tgz': ('https://ndownloader.figshare.com/files/5976018',
                '055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0'),
    'pairsDevTest.txt': ('https://ndownloader.figshare.com/files/5976009',
                        '7cb06600ea8b2814ac26e946201cdb304296262aad67d046a16a7ec85d0ff87c'),
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('.build/datasets/lfw'))
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, (url, checksum) in FILES.items():
        path = root / name
        if not path.exists():
            print(f'Downloading {name} from {url}', flush=True)
            temporary = path.with_suffix(path.suffix + '.part')
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as target:
                shutil.copyfileobj(response, target)
            if digest(temporary) != checksum:
                raise ValueError(f'Checksum mismatch for {name}; archive was not extracted')
            temporary.replace(path)
        if digest(path) != checksum:
            raise ValueError(f'Checksum mismatch for existing {name}')
        print(f'Verified {name}', flush=True)
    count = 0
    with tarfile.open(root / 'lfw.tgz', 'r:gz') as archive:
        # Extract regular JPEGs only, never archive links or arbitrary paths.
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith('.jpg'):
                continue
            target = (root / member.name).resolve()
            if not target.is_relative_to(root / 'lfw'):
                raise ValueError('Unexpected archive path')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('wb') as destination:
                shutil.copyfileobj(source, destination)
            count += 1
    manifest = {'dataset': 'LFW original (not funneled)', 'image_count': count,
                'sources': {name: {'url': url, 'sha256': checksum}
                            for name, (url, checksum) in FILES.items()}}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Ready: {count} images in {root / "lfw"}', flush=True)


if __name__ == '__main__':
    main()
