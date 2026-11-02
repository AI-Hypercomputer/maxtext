# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LingBot-World camera / action conditioning: host-side (numpy) port of `wan/utils/cam_utils.py`.

An "action sequence" is a camera trajectory: `poses.npy` ([N, 4, 4] camera-to-world matrices, OpenCV convention) and
`intrinsics.npy` ([N, 4] = fx, fy, cx, cy for the 480x832 source frames), one per video frame. It is resampled to one
pose per latent frame, made relative (frame-to-frame), turned into per-pixel Plücker rays (ray origin + direction) and
folded to the latent resolution: `[1, 6 * 8 * 8, latent_frames, latent_h, latent_w]`.
"""

import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

# Resolution the provided intrinsics refer to (the reference hard-codes 480p).
INTRINSICS_HEIGHT = 480
INTRINSICS_WIDTH = 832


def load_action(action_path: str, num_frames: int):
  """Loads `poses.npy` / `intrinsics.npy` from `action_path`; returns (c2ws, Ks, num_frames clipped to the trajectory)."""
  c2ws = np.load(os.path.join(action_path, "poses.npy"))
  intrinsics = np.load(os.path.join(action_path, "intrinsics.npy")).astype(np.float32)
  len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
  num_frames = min(((num_frames - 1) // 4) * 4 + 1, len_c2ws)
  return c2ws[:num_frames], intrinsics, num_frames


def interpolate_camera_poses(src_indices, src_rot_mat, src_trans_vec, tgt_indices) -> np.ndarray:
  """Linear translation / slerp rotation resampling of a trajectory -> [len(tgt), 4, 4] float32."""
  trans = interp1d(src_indices, src_trans_vec, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate")(
      tgt_indices
  )
  quats = Rotation.from_matrix(src_rot_mat).as_quat().copy()
  for i in range(1, len(quats)):  # no sudden sign flip of the quaternion
    if np.dot(quats[i], quats[i - 1]) < 0:
      quats[i] = -quats[i]
  rot = Slerp(src_indices, Rotation.from_quat(quats))(tgt_indices).as_matrix()
  poses = np.zeros((len(tgt_indices), 4, 4))
  poses[:, :3, :3] = rot
  poses[:, :3, 3] = trans
  poses[:, 3, 3] = 1.0
  return poses.astype(np.float32)


def se3_inverse(transforms: np.ndarray) -> np.ndarray:
  rot = transforms[:, :3, :3]
  trans = transforms[:, :3, 3:]
  rot_inv = np.swapaxes(rot, -1, -2)
  inverse = np.tile(np.eye(4, dtype=transforms.dtype)[None], (transforms.shape[0], 1, 1))
  inverse[:, :3, :3] = rot_inv
  inverse[:, :3, 3:] = -rot_inv @ trans
  return inverse


def compute_relative_poses(c2ws: np.ndarray, framewise: bool = True, normalize_trans: bool = True) -> np.ndarray:
  """Poses relative to the first frame (or frame-to-frame), translations scaled to a max norm of 1."""
  relative = se3_inverse(c2ws[0:1]) @ c2ws
  relative[0] = np.eye(4, dtype=c2ws.dtype)
  if framewise:
    relative[1:] = se3_inverse(relative[:-1]) @ relative[1:]
  if normalize_trans:
    translations = relative[:, :3, 3]
    max_norm = np.linalg.norm(translations, axis=-1).max()
    if max_norm > 0:
      relative[:, :3, 3] = translations / max_norm
  return relative


def transform_intrinsics(intrinsics: np.ndarray, height: int, width: int) -> np.ndarray:
  """Rescales [N, 4] (fx, fy, cx, cy) from the 480x832 source frames to height x width (no crop)."""
  scale = np.array([width / INTRINSICS_WIDTH, height / INTRINSICS_HEIGHT] * 2, dtype=np.float32)
  return (intrinsics * scale).astype(np.float32)


def plucker_embeddings(c2ws: np.ndarray, intrinsics: np.ndarray, height: int, width: int) -> np.ndarray:
  """Per-pixel Plücker rays [F, height, width, 6] = (ray origin, unit ray direction) in float32."""
  num_frames = c2ws.shape[0]
  ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
  i = xs.reshape(-1)[None] + 0.5  # [1, h*w]
  j = ys.reshape(-1)[None] + 0.5
  fx, fy, cx, cy = (intrinsics[:, k : k + 1] for k in range(4))  # [F, 1]
  directions = np.stack([(i - cx) / fx, (j - cy) / fy, np.ones_like(i - cx)], axis=-1)  # [F, h*w, 3]
  directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
  rays_d = directions @ np.swapaxes(c2ws[:, :3, :3], -1, -2)
  rays_o = np.broadcast_to(c2ws[:, None, :3, 3], rays_d.shape)
  return np.concatenate([rays_o, rays_d], axis=-1).reshape((num_frames, height, width, 6)).astype(np.float32)


def camera_condition(
    c2ws: np.ndarray,
    intrinsics: np.ndarray,
    height: int,
    width: int,
    latent_height: int,
    latent_width: int,
    latent_frames: int,
    chunk_size: int,
    motion_scale: float = 1.0,
) -> np.ndarray:
  """Plücker conditioning [1, 6 * s * s, latent_frames, latent_h, latent_w] (s = height // latent_h), float32.

  Mirrors `WanI2VFast.generate`: one pose per latent frame (linear / slerp resampling), frame-to-frame relative poses
  with normalized translations (optionally scaled by `motion_scale`), intrinsics rescaled to the output resolution.
  """
  ks = transform_intrinsics(intrinsics, height, width)[0]
  len_c2ws = len(c2ws)
  num_poses = (len_c2ws - 1) // 4 + 1
  num_poses = num_poses - (num_poses % chunk_size)
  if num_poses != latent_frames:
    raise ValueError(f"{len_c2ws} poses give {num_poses} latent frames, expected {latent_frames}.")
  c2ws_infer = interpolate_camera_poses(
      src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
      src_rot_mat=c2ws[:, :3, :3],
      src_trans_vec=c2ws[:, :3, 3],
      tgt_indices=np.linspace(0, len_c2ws - 1, num_poses),
  )
  c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
  c2ws_infer[:, :3, 3] *= motion_scale
  ks = np.tile(ks[None], (num_poses, 1))
  plucker = plucker_embeddings(c2ws_infer, ks, height, width)  # [f, h, w, 6]
  s_h, s_w = height // latent_height, width // latent_width
  # 'f (h c1) (w c2) c -> (f h w) (c c1 c2)' then 'b (f h w) c -> b c f h w'
  plucker = plucker.reshape((num_poses, latent_height, s_h, latent_width, s_w, 6))
  plucker = np.transpose(plucker, (5, 2, 4, 0, 1, 3)).reshape((6 * s_h * s_w, num_poses, latent_height, latent_width))
  return plucker[None]
