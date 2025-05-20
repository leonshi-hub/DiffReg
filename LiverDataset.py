import os
import numpy as np
import torch
from torch.utils.data import Dataset
import vtk


def read_vtk_points(filename):
    if filename.endswith(".vtk"):
        reader = vtk.vtkGenericDataObjectReader()
    elif filename.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()
    points = data.GetPoints()
    return np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())], dtype=np.float32)


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    N, _ = xyz.shape
    centroids = torch.zeros(npoint, dtype=torch.long).to(device)
    distance = torch.ones(N).to(device) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


class LiverDataset(Dataset):
    def __init__(self, root_dir, num_points=1024, preload=False):
        self.root_dir = root_dir
        self.num_points = num_points
        self.sample_dirs = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f)) and
               os.path.exists(os.path.join(root_dir, f, "liver_surface_partial_cam_noisy_f0.vtp"))
        ]
        self.sample_dirs.sort()
        self.preload = preload
        if preload:
            self.cache = [self._load_sample(p) for p in self.sample_dirs]
        else:
            self.cache = None

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample = self.cache[idx] if self.preload else self._load_sample(self.sample_dirs[idx])
        return sample

    def _load_sample(self, folder_path):
        preop = read_vtk_points(os.path.join(folder_path, "liver_volume_f0.vtk"))
        introp = read_vtk_points(os.path.join(folder_path, "liver_volume_f1.vtk"))
        target = read_vtk_points(os.path.join(folder_path, "liver_volume_f1.vtk"))

        if preop.shape != target.shape:
            raise ValueError(f"Mismatch in point count between f0 and f1 in {folder_path}")

        displacement = target - preop

        # === 标准化 displacement ===
        mean = displacement.mean(axis=0, keepdims=True)
        std = displacement.std(axis=0, keepdims=True) + 1e-6
        displacement_norm = (displacement - mean) / std

        # 转为 Tensor
        preop = torch.from_numpy(preop).float()
        introp = torch.from_numpy(introp).float()
        displacement = torch.from_numpy(displacement_norm).float()
        mean = torch.from_numpy(mean).float()
        std = torch.from_numpy(std).float()

        # === 统一采样或填充为 num_points ===
        def process_points(pc, n):
            if pc.shape[0] > n:
                idx = farthest_point_sample(pc, n)
                return pc[idx]
            elif pc.shape[0] < n:
                pad_idx = torch.randint(0, pc.shape[0], (n - pc.shape[0],), dtype=torch.long)
                return torch.cat([pc, pc[pad_idx]], dim=0)
            return pc

        preop = process_points(preop, self.num_points)
        introp = process_points(introp, self.num_points)
        displacement = process_points(displacement, self.num_points)
        mean = mean.expand(self.num_points, 3)
        std = std.expand(self.num_points, 3)

        return {
            'preop': preop,                 # [N, 3]
            'introp': introp,               # [N, 3]
            'displacement': displacement,   # 标准化后 [N, 3]
            'disp_mean': mean,              # [N, 3]
            'disp_std': std,                # [N, 3]
            'folder': os.path.basename(folder_path)
        }
