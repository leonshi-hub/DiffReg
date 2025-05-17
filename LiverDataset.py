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
        preop = read_vtk_points(os.path.join(folder_path, "liver_volume_f0.vtk"))        # [N, 3]
       # introp = read_vtk_points(os.path.join(folder_path, "liver_surface_partial_noisy_f1.vtp"))
        introp = read_vtk_points(os.path.join(folder_path, "liver_volume_f1.vtk"))
        target = read_vtk_points(os.path.join(folder_path, "liver_volume_f1.vtk"))

        if preop.shape != target.shape:
            raise ValueError(f"Mismatch in point count between f0 and f1 in {folder_path}")

        displacement = target - preop  # [N, 3]

        preop = torch.from_numpy(preop)
        introp = torch.from_numpy(introp)
        displacement = torch.from_numpy(displacement)

        # 对 preop & displacement 做 FPS 或填充
        if preop.shape[0] > self.num_points:
            idx_fps = farthest_point_sample(preop, self.num_points)
            preop = preop[idx_fps]
            displacement = displacement[idx_fps]
        elif preop.shape[0] < self.num_points:
            pad_size = self.num_points - preop.shape[0]
            pad_idx = torch.randint(0, preop.shape[0], (pad_size,), dtype=torch.long)
            preop = torch.cat([preop, preop[pad_idx]], dim=0)
            displacement = torch.cat([displacement, displacement[pad_idx]], dim=0)

        # 对 introp 做同样处理
        if introp.shape[0] > self.num_points:
            idx_fps = farthest_point_sample(introp, self.num_points)
            introp = introp[idx_fps]
        elif introp.shape[0] < self.num_points:
            pad_size = self.num_points - introp.shape[0]
            pad_idx = torch.randint(0, introp.shape[0], (pad_size,), dtype=torch.long)
            introp = torch.cat([introp, introp[pad_idx]], dim=0)

        return {
            'preop': preop,  # encoder_input
            'introp': introp,  # decoder_input
            'displacement': displacement,
            'folder': os.path.basename(folder_path)
        }
