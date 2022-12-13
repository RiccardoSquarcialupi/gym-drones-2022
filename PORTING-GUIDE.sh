#COMMAND for porting:
	#create env
	conda create -n battle-exp python=3.7
	#activate env
	conda activate battle-exp
	#install require lib, wait for conda to install everything(5/10 min required)
	conda install -c conda-forge ray-rllib libgcc numpy pytorch pybullet
	pip install poetry
	#build the project
	./build_project.sh
	#upgrade some lib
	pip install --upgrade numpy scipy Pillow matplotlib cycler gym pybullet torch 'ray[rllib]' stable-baselines3 tensorboard libgcc
	#file "GLIBCXX_3.4.29 not found inside /usr/lib/x86_64 etc..., 
	#the file libstdc++ of our linux distribution it's not update
	#we install the library that contain this file inside conda and export his /lib folder to LD_LIBRARY to bypass this problem 
	export LD_LIBRARY=$LD_LIBRARY_PATH:/home/SVSstudentsX/anaconda3/lib
	#double check the python version with python --version
	
