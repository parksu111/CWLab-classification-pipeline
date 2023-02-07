#  Skynet v2
This is an updated version of [Skynet](), the automatic sleep state classification pipeline made for Chung-Weber Lab.

## Table of Contents
- [Downloading the necessary files]
- [Preparing the conda environment]
- [Running Skynet]

## Downloading the necessary files
1. Download the code in this repository and unzip the folder.
2. Download the pretrained model weights titled 'best.pt' from this [page](https://drive.google.com/drive/folders/1tMhWEJwJuFSEvhqMSzvXg4wtxlxR00qm?usp=sharing).

## Preparing the conda environment
If you have been using functions in *sleepy.py*, you probably already have a conda environment ready for use.
Here, we will create a new environment to ensure that there are no conflicts between different versions of python modules.

1. Open the 'environment.yml' file in the project folder using a text editor. On the first line, change the name to whatever you wish to name the new environment. The default name is 'skynet'. Be careful not to change anything else and save the file.
2. Open the Anaconda Prompt (or terminal on Mac/Linux).
3. Navigate to the project folder using the 'cd' command as shown below:
```
cd $path/to/project/folder
```
This is an example using my linux machine. I use the 'cd' command to change the terminal's working directory to '/Desktop/skynet'. The 'ls' command lists the files in the current working directory.

4. Use the command below to create the new environment. When prompted, type 'y' and press 'enter'.
```
conda env create -f environment.yml
```
Unlike before, all the necessary packages will be automatically installed.
You can activate the new environment with the following command:
```
conda activate $name_of_new_environment
```

## Running Skynet
1. Create a folder and place the recordings you wish to classify in that folder. Make sure the folder doesn't contain any other files.
2. Open the 'config.yml' file in the project folder using a text editor and change the entry of 'recordings' to the path of the folder you made in step 1. Change the entry of 'best_model' to the path of 'best.pt' you downloaded. Don't forget to save the file.
3. Activate the conda environment we installed above.
4. Use the following command to run skynet and wait.
```
python skynetv2.py
```
5. Once the classification is done, new remidx files will automatically be generated and saved in the folders of each recording.

