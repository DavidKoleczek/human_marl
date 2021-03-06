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
│   │   └── train_agents.py              <- training "optimal" agents
│   ├── environments/
│   │   └── gridworld_environment.py
│   ├── examples/
│   │   └── manual_gridworld.py
│   ├── sharedautonomy/
│   │   └── baseline.py
│   ├── utils/
│   │   └── evaluate_policies.py
│   └── evaluate_shared_autonomy.py    <- baseline results for Shared Autonomy (Berkeley)
│
├── plots/                             <- store final plots showcasing results here
│
└── .pep8                              <- adjusting autopep8 Python styling
```
[Link](https://tree.nathanfriend.io/?s=(%27optiNs!(%27fancy!true~fullPath!false~trailMSlash!true)~J(%27J%270srcR*7Uhuman4qlearnM4traD_7s6F*3traDT%5C%27optimal%5C%27%207s0*EUH_EKexampleUmanual_HKsGa95LKutilUIpoliciesKIsG_a9623LCfor%20SG%20A9%20%7BBerkeley%7D0plots%2FFF2%203stoQ%20fDal%20ploOshowcasMCheQ0.pep8FF2*3adjustTautopep8%20PythN%20stylM0%27)~versiN!%271%27)*%20%200%5Cn2**3%3C-%204tabular_76025R26.py7agent9utNomyC%20QsulODinEenvirNmentF222GhaQdHgridworldIevaluate_Jsource!K60*LbaselDeMDgNonOts%20QreR%2F0TM%20Us5%01UTRQONMLKJIHGFEDC97654320*) to update tree manually 



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
