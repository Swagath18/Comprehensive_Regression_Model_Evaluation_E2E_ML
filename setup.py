
from setuptools import find_packages,setup #setuptools helps to build and distribute Python packages
from typing import List #The List represents a list of objects of a certain type
HYPEN_E_DOT='-e .' #To build packages while installing the requirements.txt we enalble that using -e . in requirments.txt which triggers setup.py

#function to install all libraries  
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements as shown below in setup() 
    '''
    requirements=[] #creating list
    with open(file_path) as file_obj:
        #Reading libraries from requirements.txt file
        requirements=file_obj.readlines()

        #even \n will be read, inorder to eliminate \n which is next line in requirment.txt file we will replace with blank
        requirements=[req.replace("\n","") for req in requirements]

        # we will remove -e . from requirments.txt because we dont want it in setup as it will be automatically triggered.
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

# This setup has information of entire project
setup(

name= 'ML_project',
version='0.0.1',
author='Swagath Babu',
author_email='swagathb2826@gmail.com',
Packages=find_packages(), #automatically discovers and lists all the packages in a directory.
#install_requires=['numpy', 'seaborn', 'pandas', 'matplotlib'] 
#instead of this we call a function with all the modules in one file by passing the file as argument.
install_requires=get_requirements('requirements.txt')

)
