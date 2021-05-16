# Cooperative Multi Agent Reinforcement Learning with Human in the Loop


# Installation (WIP)
## Env for Super Mario 
gym-super-mario-bros: https://pypi.org/project/gym-super-mario-bros/  
Use 'pip install gym-super-mario-bros' to install  

## Run
Run the below script in src/examples on gypsum:  


### Baseline(alpha = 0, 0.2, 0.4, 0.6, 0.8, 1)  
#!/bin/bash    
#SBATCH --partition=titanx-long  
#SBATCH --gres=gpu:1  
#SBATCH --output=alpha_0.1.txt  
python sm_alpha_gypsum.py --discount_factor 0.9 --max_timesteps 5000000 --final_exploration 0.05 --pilot_name "noisy_pilot" --num_models 5 --alpha 0.1

### Budget(budget = 0, 50, 100, 150, 200, 250, 300, 350, 400)  
#!/bin/bash    
#SBATCH --partition=titanx-long  
#SBATCH --gres=gpu:1  
#SBATCH --output=budget_50.txt  
python sm_budget_gypsum.py --discount_factor 0.9 --max_timesteps 5000000 --final_exploration 0.05 --pilot_name "noisy_pilot" --num_models 5 --budget 50


### Penalty(budget = 0.05, 0.1, 0.2, 0.5, 1, 2, 5)  
#!/bin/bash    
#SBATCH --partition=titanx-long  
#SBATCH --gres=gpu:1  
#SBATCH --output=penalty_0.1.txt  
python sm_penalty_gypsum.py --discount_factor 0.9 --max_timesteps 5000000 --final_exploration 0.05 --pilot_name "noisy_pilot" --num_models 5 --intervention_punishment 0.1


### Penalty_adapt(intervention_rate = 0, 0.2, 0.4, 0.6, 0.8, 1)  
#!/bin/bash    
#SBATCH --partition=titanx-long  
#SBATCH --gres=gpu:1  
#SBATCH --output=interventio_rate_0.2.txt  
python sm_penalty_adapt_gypsum.py --discount_factor 0.9 --max_timesteps 5000000 --final_exploration 0.05 --pilot_name "noisy_pilot" --num_models 5 --intervention_rate 0.2