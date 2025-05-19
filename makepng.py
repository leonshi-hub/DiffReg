import open3d as o3d
import numpy as np
import sys

def load_txt_pointcloud(path):
    pts = np.loadtxt(path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python vis_pointcloud.py preop.txt introp.txt warped.txt")
        sys.exit(1)

    preop_path, introp_path, warped_path = sys.argv[1:4]
    preop = load_txt_pointcloud(preop_path)
    introp = load_txt_pointcloud(introp_path)
    warped = load_txt_pointcloud(warped_path)

    preop.paint_uniform_color([1, 0, 0])    # 红色
    introp.paint_uniform_color([0, 1, 0])   # 绿色
    warped.paint_uniform_color([0, 0, 1])   # 蓝色

    o3d.visualization.draw_geometries([preop, introp, warped])