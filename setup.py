from setuptools import find_packages
from setuptools import setup

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

def get_install_requires():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f.readlines() if not line.startswith("-")]


setup(
    name='pyCtrlLoopNoise',
    version='0.0.1',
    url='url="https://github.com/kaikai-liu/pyCtrlLoopNoise',
    license='',
    author='Kaikai Liu',
    author_email='kaikailiu@ucsb.edu',
    description='Photonics simulation tools',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=get_install_requires(),
    python_requires=">=3.6"
)