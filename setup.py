#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

# Version number
version = '1.0'
setup(name = 'AAGP',
      version = version,
      packages = find_packages(),
      install_requires=[
          'matplotlib>=3.5.3','pandas>=1.3.5','scipy>=1.7.3','xgboost>=1.6.2',
          'scikit-learn>=1.0.2','joblib==1.2.0','numpy>=1.21.6','tqdm==4.65.0','seaborn==0.12.2',
            'GPy==1.10.0'
      ],
      classifiers=[
                   # 'Natural Language :: English',
                   # 'Operating System :: MacOS :: MacOS X',
                   # 'Operating System :: Microsoft :: Windows',
                   # 'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   ]
      )
