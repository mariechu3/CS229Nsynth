# NSynth
This repository contains material based on [Naddim Kawwa's](https://github.com/NadimKawwa/NSynth) Udacity's Machine Learning Engineer Nando Degree (MLEND) capstone project. The objective of this project is to classify wave files (.wav) based on their instrument family. 

# Run on a GCP instance

Go to GCP and choose compute optimized instance Series 'C2'. Make sure to change the Boot disk size to something larger (like 100GB). After the instance is started click on 'SSH' to connect.

```
# For train
wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz && tar -xf nsynth-train.jsonwav.tar.gz 
# For test
wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz && tar -xf nsynth-test.jsonwav.tar.gz 
sudo apt-get install python3-pip -y
wget http://repo.continuum.io/archive/Anaconda3-4.0.0-Linux-x86_64.sh
bash Anaconda3-4.0.0-Linux-x86_64.sh
```
Type "yes's" and continue

```
source ~/.bashrc
jupyter notebook --generate-config
vim ~/.jupyter/jupyter_notebook_config.py
```

Add these lines to config.py at the top
```
c = get_config()
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 5000
```
Follow [this](https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52) to set up the port to 5000 

```
sudo apt install git -y
git clone https://github.com/mariechu3/CS229Nsynth.git
cd CS229Nsynth
mkdir SpectroImages
cd SpectroImages
mkdir train
cd ../../
pip3 install pandas
pip3 install matplotlib
pip3 install librosa
sudo apt install tmux
```
create a tmux session (ctrl+b d (to detach tmux session))
```
tmux
#choose your own password
jupyter notebook password
jupyter-notebook --no-browser --port=5000
```

Open notebook at http://<Public IP address>:5000 (i.e. http://34.152.60.31:5000)

Run notebook:
    - import statements
    - change train/test/valid path to the correct path and run it
    - run feature_extract cell
    - run the cell labeled "Test, Train, or Valid" and the correct splits (github has the example for train)

cd CS229Nsynth/SpectroImages/train/
ls -1 |wc -l