from setuptools import find_packages
from setuptools import setup

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
    description='Control loop noise analysis',
    packages=find_packages(),
    install_requires=get_install_requires(),
    python_requires=">=3.6"
)
