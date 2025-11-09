#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt.models.vggt import VGGT

def main():
    parser = argparse.ArgumentParser(description="VGGT-SLAM on TUM dataset with point cloud output")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to TUM RGB folder (e.g., .../rgb)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--max_loops", type=int, default=1)
    parser.add_argument("--min_disparity", type=float, default=50)
    parser.add_argument("--conf_threshold", type=float, default=25)
    parser.add_argument("--submap_size", type=int, default=16)
    parser.add_argument("--tum", action="store_true", help="Use TUM naming convention for output files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        use_point_map=False,
        use_sim3=False,
        gradio_mode=False,
        vis_stride=2,
        vis_point_size=0.003,
    )

    print("Loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval().to(device)

    print(f"Loading images from {args.image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*"))
                   if "depth" not in os.path.basename(f).lower()
                   and "txt" not in os.path.basename(f).lower()
                   and "db" not in os.path.basename(f).lower()]
    image_names = utils.sort_images_by_number(image_names)
    print(f"Found {len(image_names)} images")

    if len(image_names) == 0:
        print("❌ No images found. Check your --image_folder path.")
        return

    image_subset = []
    for image_name in tqdm(image_names):
        img = cv2.imread(image_name)
        enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, False)
        if enough_disparity:
            image_subset.append(image_name)

        if len(image_subset) == args.submap_size or image_name == image_names[-1]:
            print(f"Processing submap with {len(image_subset)} frames...")
            predictions = solver.run_predictions(image_subset, model, args.max_loops)
            solver.add_points(predictions)
            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)
            image_subset = []

    print("✅ Reconstruction complete.")
    print(f"Submaps: {solver.map.get_num_submaps()}, Loops: {solver.graph.get_num_loops()}")

    # === 保存轨迹和点云 ===
    if args.tum:
        # 取上级文件夹名，如 rgbd_dataset_freiburg2_desk_with_person
        dataset_name = os.path.basename(os.path.dirname(args.image_folder))
    else:
        dataset_name = os.path.basename(args.image_folder)

    traj_path = os.path.join(args.output_dir, f"traj_{dataset_name}.txt")
    ply_path = os.path.join(args.output_dir, f"{dataset_name}.ply")

    print(f"Saving trajectory: {traj_path}")
    solver.map.write_poses_to_file(traj_path)

    print(f"Saving dense point cloud: {ply_path}")
    solver.map.write_points_to_file(ply_path)

    print("All results saved successfully.")

if __name__ == "__main__":
    main()
