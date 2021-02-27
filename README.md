# Cooperative Multi Agent Reinforcement Learning with Human in the Loop

# Code Structure

```
.
├── src/
│   ├── agents/
│   │   ├── humantabular_agent.py
│   │   └── qlearningtabular_agent.py
│   ├── environments/
│   │   └── gridworld_environment.py
│   ├── examples/
│   │   └── gridworld_environment.py
│   ├── sharedautonomy/
│   │   └── baseline.py
│   ├── utils/
│   │   └── evaluate_policies.py
│   └── evaluate_shared_autonomy.py           <- baseline results for Shared Autonomy (Berkeley)
└── .pep8                                     <- adjusting autopep8 Python styling
```
[Link](https://tree.nathanfriend.io/?s=(%27optiJs!(%27fancy!true~fullPath!false~trailGSlash!true)~I(%27I%27Fsrc4agKLhuman3*qlearnG362example2s9a54*E7utilLDpolicies7Ds9_a5.py**HE%20results%20for%20S9%20A5%20%7BBerkeley%7DF.pep8CCCCCHadjustG%20autopep8%20PythJ%20stylG0%27)~versiJ!%271%27)*%20%200F*2Lgridworld_673tabular_agK74%2F05utJomy6envirJmK7.py09haredC***Devaluate_EbaselineF%5CnGingH%3C-%20Isource!JonKentLs4*%01LKJIHGFEDC97654320*) to update tree manually 



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