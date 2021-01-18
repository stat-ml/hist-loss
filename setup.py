import os
import setuptools

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

from histloss import __version__

def load_requirements(filename):
    with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
        return f.read().splitlines()

setuptools.setup(
    name="hist-loss",
    version=__version__,
    author="Maxim Panov and Nikita Mokrov and Roman Lisov",
    author_email="nikita.mokrov@skoltech.ru",
    description="Package with losses for distribution learning",
    url="https://github.com/stat-ml/histloss",
    packages=setuptools.find_packages(),
    license="Apache License 2.0",
    install_requires=load_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
)