#COMMAND for porting:
	#create env
	conda create -n battle-exp
	conda activate battle-exp
	conda install python=3.7
	#wait for conda to install everything(5/10 min required)
	#activate env 	
	#install all the required lib
	conda install -c conda-forge ray-rllib libgcc poetry numpy pytorch pybullet
	#build the project
	./build_project.sh
	#upgrade some lib
	#pip3 install --upgrade numpy Pillow matplotlib cycler
	#pip3 install --upgrade gym pybullet stable_baselines3 'ray[rllib]'
	pip install --upgrade numpy scipy Pillow matplotlib cycler gym pybullet torch pytorch 'ray[rllib]' stable-baseline3 tensorboard
	#file "GLIBCXX_3.4.29 not found inside /usr/lib/x86_64 etc..., 
	#the file libstdc++ of our linux distribution it's not update we install the library 	
	#contains this file inside conda and export his /lib folder to LD_LIBRARY to bypass this problem 
	export LD_LIBRARY=$LD_LIBRARY_PATH:/home/SVSstudentsX/anaconda3/lib
	
	#double check the python version with python --version
	
