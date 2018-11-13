[//]: # (Image References)

[image1]: reacher.dms "Trained Agent" 
![Trained Agent][image1]

# DDPG ContinuousControl
Deep Deterministic Policy Gradient for Continuous Control


### Introduction

This repository contains a simple Neural Network driven DDPG algorithm running in Unity ML Agent. This model is sometimes called "Deep Deterministic Policy Gradient", but this is a misnomer since there is usually nothing deep in the model architecture. Typically Deep network are convolutional networks models using hundreds or thousands of layers. Our model uses a Multi Layer Perceptron with two small hidden layers of 64 nodes on each hidden layer.

The model is implemented in Python 3 using PyTorch.

Using this model, we will train 20 agents driving each a robotic arm to push around a ball in a square world represented in Unity ML.  


# Goal

  Thus, the goal of the 20 agents is to push the ball while maintaining control over it.

# Rewards

A reward of +1 is provided for keeping the ball in arm's reach.
A reward of -1 is provided for loosing control over the ball.

# State space

The state space has 33 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions. 

# Action space

 The state space is continuous and its scope encompasses the various positions the robotic arm can take.

The task is episodic, and in order to solve the environment, the agent must get an average score of +80 over 100 consecutive episodes.

### Getting Started

1. Install Unity ML https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md


2. Download the Unity environment from one of the links below.  You need only select the environment that matches your operating system:
     
Mac: "path/to/Reacher.app"
Windows (x86): "path/to/Reacher_Windows_x86/Reacher.exe"
Windows (x86_64): "path/to/Reacher_Windows_x86_64/Reacher.exe"
Linux (x86): "path/to/Reacher_Linux/Reacher.x86"
Linux (x86_64): "path/to/Reacher_Linux/Reacher.x86_64"
Linux (x86, headless): "path/to/Reacher_Linux_NoVis/Reacher.x86"
Linux (x86_64, headless): "path/to/Reacher_Linux_NoVis/Reacher.x86_64"
For instance, if you are using a Mac, then you downloaded Reacher.app. If this file is in the same folder as the notebook, then the line below should appear as follows:
env = UnityEnvironment(file_name="Reacher.app")
    
3. Place the file in the Continuous Control folder, and unzip (or decompress) the file. 

4. Install Anaconda3.

    Before you begin, you should have a non-root user with sudo privileges set up on your computer.

    The best way to install Anaconda is to download the latest Anaconda installer bash script, verify it, and then run it.

    Find the latest version of Anaconda for Python 3 at the Anaconda Downloads page.
    https://www.anaconda.com/download/#macos
    Install ANACONDA3 following instructions from the anaconda web site.
    
5. Create a virtual environment for python    
    
    In a terminal type: 

    conda create -n drlnd python=3.6
    
    source activate drlnd
    
    To stop the virtual environment once you are done, type deactivate in the terminal.
    
    Always start the drlnd virtual environment before starting the jupyter notebook or the python script,
    else you will get errors when running the code.
    


6. Install dependencies
    
    cd python

    python3 setup.py install -r requirements.txt
    

7. Start the notebook 
    
    cd ../
    
    jupyter notebook

8. Run the agent

   To start training, simply open ContinuousControl.ipynb in Jupyter Notebook and follow the instructions given there.
   You can either train the agent: this takes about 3 hours to reach a score of 80, 48 hours to run 1800 iterations, or you can skip the training and watch a trained agent using the provided trained weights.



