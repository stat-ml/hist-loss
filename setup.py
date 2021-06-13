import os
import setuptools

from histloss import __version__

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements(filename):
    with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
        return f.read().splitlines()


with open('README.md', 'r') as f:
    DESCRIPTION = f.read()

setuptools.setup(
    name="hist-loss",
    version=__version__,
    author="Maxim Panov and Nikita Mokrov and Roman Lisov",
    author_email="nikita.mokrov@skoltech.ru",
    description="Package with losses for distribution learning",
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
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
