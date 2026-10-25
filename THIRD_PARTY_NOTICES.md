# Third-party materials

Original project source and modifications are released under Apache-2.0; see LICENSE.
Separate third-party data and models retain their own terms.

- The patches in patches/ modify NVIDIA video_to_data source at
  7eb9721088615fe9201634a6be5cf48b049d1090. The mesh-refresh patch changes
  regenerated mesh handling; the orientation patch adds an opt-in objective and tests.
  Upstream: https://github.com/nvidia-isaac/video_to_data/tree/7eb9721088615fe9201634a6be5cf48b049d1090
  Source license: Apache-2.0; upstream documentation/skills: CC-BY-4.0.
  Preserve applicable upstream notices when redistributing further source.
- Sequence identifiers and derived job lists refer to NVIDIA's Video to Data
  Challenge Dataset: https://huggingface.co/datasets/nvidia/video_to_data_challenge
  Released data is CC-BY-4.0: https://creativecommons.org/licenses/by/4.0/
  Project recipes, retargeting, contact labels, reconstructions and annotations
  are project modifications; NVIDIA does not endorse this entry.
- The official submission kit is downloaded separately, not vendored here.
- SAM 3D Body/MHR, CARI4D, SOMA-X and other pretrained weights are downloaded
  separately under their respective terms. This repository's license does not
  relicense those materials or grant gated-model access.
