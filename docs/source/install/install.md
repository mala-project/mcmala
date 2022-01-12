# Installation of MC-MALA

## Python packages

In order to run MC-MALA you have to have the following packages installed:

* ase
* numpy

See also the `requirements.txt` file.
You can install each Python package with

```sh
$ pip install packagename
```

or all with

```sh
$ pip install -r requirements.txt
```

Further, you need to have MALA installed if you want to use MC-MALA to run
a MALA simulation. MALA is not included in the requirements (yet), because it
is not yet available on PyPI. If you do not install MALA, you can still
use this package for simple tests using the Ising model.
After sorting out the dependencies, you can install MC-MALA with 

```sh
$ pip install -e .
```

(note: the `-e` is absolutely crucial, so that changes in the code will be
reflected system wide)
