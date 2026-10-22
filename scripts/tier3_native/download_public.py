"""GPU-box-only anonymous public downloads; never reads a Mac HF cache."""
from pathlib import Path
import os, subprocess, urllib.request, tarfile, json, hashlib
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
os.environ['HF_HOME']=str(Path(os.environ['NATIVE_WEIGHTS'])/'hf_home')
os.environ['TORCH_HOME']=str(Path(os.environ['NATIVE_WEIGHTS'])/'sam3d_body/torch_home')
from huggingface_hub import snapshot_download
root=Path(os.environ['NATIVE_WEIGHTS'])
snapshot_download('Ruicheng/moge-2-vitl-normal',revision='b135031bae30b5ac2ae141a0e68717795ce38340',cache_dir=root/'hf_home/hub',token=False)
snapshot_download('facebook/sam2.1-hiera-large',local_dir=root/'sam2',allow_patterns=['sam2.1_hiera_large.pt','*.yaml','*.json'],token=False)
for name,rev in [('dinov3','6876159a11b4df116f30f667f8c9888617df0751'),('dinov2','7764ea0f912e53c92e82eb78a2a1631e92725fc8')]:
    dest=root/f'sam3d_body/torch_home/hub/facebookresearch_{name}_main'
    if not ((dest/'hubconf.py').is_file() and (dest/'.native_archive_pin').is_file() and (dest/'.native_archive_pin').read_text().strip()==rev):
        dest.mkdir(parents=True,exist_ok=True)
        archive=Path(os.environ['NATIVE_ROOT'])/'third_party'/f'{name}-{rev}.tar.gz'
        urllib.request.urlretrieve(f'https://codeload.github.com/facebookresearch/{name}/tar.gz/{rev}',archive)
        subprocess.run(['tar','-xzf',str(archive),'--strip-components=1','-C',str(dest)],check=True)
        (dest/'.native_archive_pin').write_text(rev+'\n')
import torch
for name in ['dinov2_vitb14','dinov2_vits14']:
    torch.hub.load(str(root/'sam3d_body/torch_home/hub/facebookresearch_dinov2_main'),name,source='local',pretrained=True)
checks={'dinov2_vitb14_pretrain.pth':'0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73','dinov2_vits14_pretrain.pth':'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'}
for filename,expected in checks.items():
    p=root/'sam3d_body/torch_home/hub/checkpoints'/filename
    actual=hashlib.sha256(p.read_bytes()).hexdigest()
    assert actual==expected, filename
for folder,run in [('12Te_3TELLes5cim1d7F7EBTwUSe7iRBj','2024-01-11-20-02-45'),('1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC','2023-10-28-18-33-37')]:
    dest=root/'foundationpose/nvlabs_pytorch'/run
    if not all((dest/name).is_file() and (dest/name).stat().st_size>0 for name in ['model_best.pth','config.yml']):
        subprocess.run(['gdown','--folder',f'https://drive.google.com/drive/folders/{folder}','-O',str(dest)],check=True)
print('PUBLIC_WEIGHTS_DONE',flush=True)
