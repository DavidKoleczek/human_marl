# On Optimizing Interventions in Shared Autonomy


# Installation 
## RL library
[The Autonomous Learning Library](https://github.com/cpnota/autonomous-learning-library/tree/31e5aa9d85b4f1d1ad386b8e87c7d09fd8d31302) version 0.6.2  
Use 'pip install autonomous-learning-library==0.6.2'  
Also install the related library including [pytorch](https://pytorch.org/) > 1.3 

## Env for Super Mario 
[gym-super-mario-bros](https://pypi.org/project/gym-super-mario-bros/)  
Use 'pip install gym-super-mario-bros' to install  

# Run
## Get a simulated agent.   
You can train your own agent with any methods 
or use the code we provide, src/examples/lunar_lander_full_pilot.py and super_mario_full_pilot.py to train. We have already provided the trained models in savedModels.

## Use the simulated agent to train copilot.  
Use the code in src/examples to train the copilot. 