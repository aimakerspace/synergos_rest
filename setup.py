#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os
from setuptools import setup

###########
# Helpers #
###########

def read_readme(fname):
    with open(
        os.path.join(os.path.dirname(__file__), fname), 
        encoding='utf-8'
    ) as f:
        return f.read()


def read_requirements(fname):
    with open(
        os.path.join(os.path.dirname(__file__), fname), 
        encoding='utf-8'
    ) as f:
        return [s.strip() for s in f.readlines()]     


setup(
    name="synergos_rest",
    version="0.1.0",
    author="AI Singapore",
    author_email='synergos-ext@aisingapore.org',
    description="REST-RPC component for the Synergos network",
    long_description=read_readme('README.md'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords="synergos horizontal vertical federated learning logging graylog",
    url="https://github.com/aimakerspace/synergos_rest.git",
    license="MIT",
    packages=[
        "rest_rpc", 
        "rest_rpc.connection", 
        "rest_rpc.connection.core",
        "rest_rpc.training", 
        "rest_rpc.training.core",
        "rest_rpc.training.core.hypertuners",
        "rest_rpc.evaluation",
        "rest_rpc.evaluation.core",        
    ],
    package_dir={'rest_prc': "rest_rpc"},
    package_data={'rest_prc': ["templates/**/*_schema.json"]},
    python_requires = ">=3.7",
    install_requires=read_requirements("requirements.txt"),
    include_package_data=True,
    zip_safe=False
)
