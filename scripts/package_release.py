"""Archive the verified portable build, excluding local models and user state."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[1]
source = root / 'dist/PureGPU3D'
target = root / 'release-assets'
target.mkdir(exist_ok=True)
archive = target / 'PureGPU3D-v1.0.2-windows-x64.zip'
manifest = {}
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as out:
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if relative.parts[0] in ('models', 'data', 'licenses'):
            continue
        name = 'PureGPU3D/' + relative.as_posix()
        out.write(path, name)
        manifest[name] = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    out.writestr('PureGPU3D/models/', '')
    out.write(root / 'README.md', 'PureGPU3D/README.md')
    out.writestr('PureGPU3D/RELEASE.txt', 'PureGPU3D v1.0.2\nPortable Windows x64 desktop build. Extract the entire folder.\n')
(target / 'build-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
with archive.open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
(target / 'SHA256SUMS.txt').write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
print(archive, archive.stat().st_size, digest)
