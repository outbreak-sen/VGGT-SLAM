 
# 环境配置


vggt-slam

~~~
git clone https://github.com/MIT-SPARK/VGGT-SLAM

cd VGGT-SLAM

conda create -n vggt-slam python=3.11

./setup.sh

pip install evo --user
~~~

# 运行方法

~~~
python demo_traj.py \
  --image_folder /share/datasets/TUM/Dynamics/rgbd_dataset_freiburg2_desk_with_person/rgb \
  --submap_size 16 \
  --max_loops 1 \
  --tum

python demo_traj.py \
  --image_folder /share/datasets/TUM/Dynamics/rgbd_dataset_freiburg2_desk_with_person/rgb \
  --submap_size 16 \
  --max_loops 1 \
  --tum

~~~