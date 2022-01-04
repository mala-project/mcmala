from setuptools import setup, find_packages

# Doing it as suggested here:
# https://packaging.python.org/guides/single-sourcing-package-version/
# (number 3)

version = {}
with open("mcmala/version.py") as fp:
    exec(fp.read(), version)

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="mcmala",
    version=version["__version__"],
    description=("Monte-Carlo fo Materials Learning Algorithms. "
                 "A Monte-Carlo frontend for MALA."),
    long_description=readme,
    long_description_content_type='text/markdown',
    url="https://github.com/mala-project/mcmala",
    author="MALA developers",
    license=license,
    packages=find_packages(exclude=("test", "docs", "examples", "install")),
    zip_safe=False,
    install_requires=open('requirements.txt').read().splitlines(),
)
