################
## setup.py   ##
################

# check with:
# (base)$ python setup.py check

from setuptools import find_packages, setup

def readme():
    try:
        with open('README.rst') as f:
            return f.read()
    except:
        return "NO README."

def read_version():
    try:
        with open('version.txt') as f:
            return f.read().strip().split("=")[1]
    except:
        return "1.0.0"

setup(

    # meta

    name="rklearn", # the pip name
    version=read_version(),
    description="The Régis Kla ML package",
    long_description=readme(),
    url="http://xxxxxx",
    author="Régis KLA",
    author_email="klaregis@gmail.com",
    license="MIT",

    # body

    zip_safe=False,
    packages = find_packages(exclude=['contrib', 'docs']),

    # DO THIS IN PARALLEL OF MANIFEST.in

    package_dir = {

        "rklearn" : "rklearn",
        "rklearn.tests" : "rklearn/tests"

    },

    # Requirements

    python_requires='>=3.0.*, <4',
    install_requires=[]

)
