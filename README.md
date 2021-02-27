# Cooperative Multi Agent Reinforcement Learning with Human in the Loop

# Code Structure
* Main executable scripts are in the top level of `src`, start here!
* Other examples are in `src/examples`
```
.
├── src/
│   ├── agents/
│   │   ├── humantabular_agent.py
│   │   ├── qlearningtabular_agent.py
│   │   └── train_agents.py             <- training "optimal" agents
│   ├── environments/
│   │   └── gridworld_environment.py
│   ├── examples/
│   │   └── manual_gridworld.py
│   ├── sharedautonomy/
│   │   └── baseline.py
│   ├── utils/
│   │   └── evaluate_policies.py
│   └── evaluate_shared_autonomy.py     <- baseline results for Shared Autonomy (Berkeley)
└── .pep8                               <- adjusting autopep8 Python styling
```
[Link](https://tree.nathanfriend.io/?s=(%27optiKs!(%27fancy!true~fullPath!false~trailLSlash!true)~I(%27I%27Hsrc56Mhuman3qlearnL3traE_6s9D*4traEL%20%5C%27optimal%5C%27%206s0C2example2sFa75*JNutilMGpoliciesNGsF_a794J%20results%20for%20SF%20A7%20%7BBerkeley%7DH.pep8DDD*4adjustL%20autopep8%20PythK%20stylL0%27)~versiK!%271%27)*%20%200H*2Mgridworld_CN3tabular_6N*4**%3C-%205%2F06agent7utKomy9.pyCenvirKmentD****EinFharedGevaluate_H%5CnIsource!JbaselEeKonLEgMs5*N90%01NMLKJIHGFEDC97654320*) to update tree manually 



# Installation (WIP)
## Create a Python Environment
### Windows 
* Create a new Python virtual environment: `path_to_python_installation -m venv path_to_where_you_want_venv ` 
  * Example: `"C:\Program Files\Python38\python" -m venv "D:\Programming\human_marl\env"`
* Active the environment:  `.\env\Scripts\activate`
* Upgrade pip: `python -m pip install --upgrade pip setuptools wheel`
* Installation of [ALL](https://github.com/cpnota/autonomous-learning-library) required a manual download of [swig](http://www.swig.org/download.html), adding the folder to path, and then `pip install autonomous-learning-library`
* NOTE: `requirements.txt` not currently up to date
* Recommended `pip install autopep8` for automatic code formatting and `pip install pylint` for code analysis


## Setup Malmö
https://github.com/Microsoft/malmo/tree/master/MalmoEnv
### Windows
* Clone malmo:  `git clone https://github.com/Microsoft/malmo.git`
* `cd malmo/Minecraft`
* Populate the version.properties file with the correct version:  `(echo | set /p dummyName="malmomod.version=" && type "..\VERSION") > ./src/main/resources/version.properties`
* Go to the MalmoEnv dir: 
  * `cd ../.. `
  * `cd malmo/MalmoEnv`
* Install the package: `py setup.py install`


## Malmo Experimentation
### Windows
#### Starting a Minecraft client
* From your project directory: `cd malmo/Minecraft`
* Launch a Minecraft client: `launchClient.bat -port 9000 -env`

#### Starting an example mission
* Go to MalmoEnv where scripts written in Python live: `cd malmo/MalmoEnv`
* Run one of the examples: `py run.py --mission missions/mobchase_single_agent.xml --port 9000 --episodes 10`
