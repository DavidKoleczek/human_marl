# Cooperative Multi Agent Reinforcement Learning with Human in the Loop


# Installation (WIP)
## Env for Super Mario 
gym-super-mario-bros: https://pypi.org/project/gym-super-mario-bros/  
Use 'pip install gym-super-mario-bros' to install  

## Run
Run the below script in src/examples on gypsum:  

#!/bin/bash  
#SBATCH --partition=titanx-long  
#SBATCH --gres=gpu:1
#SBATCH --output=noisy_pilot_5.txt  
python sp_penaltyVSintervention_gypsum.py --discount_factor 0.9 --intervention_punishment 5 --pilot_name "noisy_pilot" --num_models 3
