echo "Y" | pip uninstall gym_pybullet_drones
rm -rf dist/
poetry build 
pip install dist/gym_pybullet_drones-1.0.0-py3-none-any.whl
cd tests
python test_build.py
rm -rf results
cd ..
#require python 3.7-3.9
# pip3 install numpy scipy Pillow matplotlib cycler gym pybullet torch 'ray[rllib]' stable-baseline3 scipy tensorboard
# pip3 install --upgrade numpy scipy Pillow matplotlib cycler gym pybullet torch 'ray[rllib]' stable-baseline3 scipy tensorboard