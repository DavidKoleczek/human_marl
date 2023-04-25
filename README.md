# On Optimizing Interventions in Shared Autonomy
## Demo code available [here](https://github.com/DavidKoleczek/hitl_demo)!

# Code Structure
* Main executable scripts are in the `experiments` folder, start here!
* Library code is located under the `src` folder.

# Installation
High level installation steps:
1. Create a Python3.8 Environment
2. Install PyTorch
3. Install remaining dependencies from `requirements.txt`

### Windows 
1. Create a new Python virtual environment: `path_to_python_installation -m venv path_to_where_you_want_venv ` 
    * Example: `"C:\Program Files\Python38\python" -m venv "D:\Programming\human_marl\env"`
2. Active the environment:  `.\env\Scripts\activate`
3. Upgrade pip: `python -m pip install --upgrade pip setuptools wheel`
4. Install the correct version of PyTorch for your system at [pytorch.org](https://pytorch.org/get-started/locally/)
5. Install the other dependencies with `pip install -r requirements.txt`
6. The [following line of code](https://github.com/cpnota/autonomous-learning-library/blob/31e5aa9d85b4f1d1ad386b8e87c7d09fd8d31302/all/experiments/writer.py#L25) in ALL must be changed to something that doesn't use a colon in the file name such as `current_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S %f')`
7. Install the library code as a package `pip install --editable .`
8. For development, we recommended `pip install autopep8` for automatic code formatting and `pip install pylint` for code analysis
9. Installation of [ALL](https://github.com/cpnota/autonomous-learning-library) required a manual download of [swig](http://www.swig.org/download.html), adding the folder to path, and then `pip install autonomous-learning-library==0.6.2`

