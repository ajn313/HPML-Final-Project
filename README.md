# HPML-Final-Project

Here is the final project of HPML course, the team members are Andre Nakkab and Hao Tian.

Project description:

In this project, we hope to optimize a novel method of generative design which the Andre Nakkab developed recently. Here is the past work repo : https://github.com/ajn313/Deep-Learning-Final-Project-Fall2021.

This architecture, which we will refer to as the Targeted Generative Adversarial Network (TGAN) utilizes a fully trained image recognition network as a sort of tertiary element in a generative adversarial network. This new architecture is meant to generate specific, convincing categorical elements based on the image set used for training. 

Repository structure:

All our code files are in the "project files" folder, you can upload the "HPML_project_script.py" to the NYU Greene HPC, then run with sbatch file.
For example, for 1 gpu running, we can use the command: `sbatch project_sbatch_1_gpu.sbatch`.

All our output pictures include the measurement of loss and time are in other folders, you can check those results.



