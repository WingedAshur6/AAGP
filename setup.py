#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, packages
import numpy

# Version number
version = '1.0'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name = 'AAGP',
      version = version,
      author = read('AUTHORS.txt'),
      description = ("Adjacency-Adaptive Gaussian Process"),
      license = "BSD 3-clause",
      keywords = "gaussian process",
      url = "",
      long_description=read('README.md'),
      packages = packages.find_packages(),
      install_requires=[
          'matplotlib>=3.5.3','pandas>=1.3.5','scipy>=1.7.3','xgboost>=1.6.2',
          'scikit-learn>=1.0.2','joblib==1.2.0','numpy>=1.21.6','tqdm==4.65.0','seaborn==0.12.2',
          # 'deepgp@https://github.com/SheffieldML/PyDeepGP.git#egg=deepgp',
      ],
      include_dirs=[numpy.get_include()],
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   ]
      )
