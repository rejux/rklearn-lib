# The Developer's Guide

## 1. Introduction

This document is the mini guide for the Python developer that contributes to the *rklearn-lib* code.

## 2. The Virtual Environment

It is highly advised to work into a virtual environment. Execute the following commands to prepare it: 

```shell
(base) $ sudo apt-get update
...
(base) $ python3.6 -m venv ~/venvs/rklearn-lib
(base) $ source ~/venvs/rklearn-lib/bin/activate 
(rklearn-lib) (base) $ python --version
Python 3.6.9
```

**Note:**

* Notice the version of Python we use here: *3.6*. This details is important if you use old CPUs that don't support AVX and AVX2 instruction sets. If you have a new CPU, then you can safely use *python* generic name and the last version will be used. More details can be found in the *Troubleshooting* section.

Optionally, one can upgrade to the last version of *pip* compatible with the running Python version:

```shell
(rklearn-lib) (base) $ pip install --upgrade pip
```

Update the virtual environment with the required packages:

```shell
(rklearn-lib) (base) $ pip install -r requirements_tf_v1.txt
```

**Note:**

* After a session with the container - generally new packages were installed - always generate the *requirements_xxx.txt* file: 

```shell
(rklearn-lib)...$ pip freeze >requirements_tf_v1.txt
```

* If you use the CDE, as an host's user (e.g. outside the container) you should commit the container t oa new tag:

```shell
$ sudo docker commit rklearn-lib-cde rklearn-lib-cde:x.y.z
```

## 3. Develop

This section presents the daily work for the developer.

**Prerequisite:**



In order to be in *real* usage conditions, it is advised to install the library as a normal user, during your developmnt sessions:

```shell
(rklearn-lib) (base) $ pip install --upgrade -e .
```

or 

```shell
(rklearn-lib) (base) $ python setup.py develop --upgrade
```

### 3.1. Testing

The tests are grouped either as *unit tests (ut)* or *integration tests (it)*: 

```shell

```


**Unit Tests**





## 4. Troubleshooting

**Older CPU and Tensorflow:**

Test if TF can run safely on your CPU:

```shell
(rklearn-lib) (base) ...$ python
...
>>> import tensorflow
Illegal instruction (core dumped)
```

This problem is generally caused by the missing of AVX and AVX2 instruction sets support. Thus, let us first check that we effectively miss the AVX support: 

```shell
$ more  /proc/cpuinfo | grep flags |grep avx
```

If notthing is displayed, then your CPU does not support the AVX instruction set. And thus, TF >= 1.5 cannot work properly. The solutions are (1) downgrade tensorflow to 1.5 version, or (2) build tensorflow from source, or (3) download and install a precompiled version without AVX.

We choose the option (3) and follow instruction [here](https://github.com/yaroslavvb/tensorflow-community-wheels/issues/103).

```shell
(...) $ pip install https://github.com/mdsimmo/tensorflow-community-wheels/releases/download/1.13.1_cpu_py3_6_amd64/tf_nightly-1.13.1-cp36-cp36m-linux_x86_64.whl
```

Retest the import of tensorflows (e.g. it must import it silently):

```shell
$ python -c "import tensorflow"
```

**Note:** Notice that this version of tensorflow is compatible with *numpy==1.16.0*. 

**END OF DOCUMENT.**
