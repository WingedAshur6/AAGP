#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages
import numpy

# Version number
version = '1.0'
setup(name = 'AAGP',
      version = version,
      packages = find_packages(),
      install_requires=[
          'matplotlib>=3.5.3','pandas>=1.3.5','scipy>=1.7.3','xgboost>=1.6.2',
          'scikit-learn>=1.0.2','joblib==1.2.0','numpy>=1.21.6','tqdm==4.65.0','seaborn==0.12.2',
      'PyDeepGP'
      ]
      dependency_links = [
            'git+https://github.com/SheffieldML/PyDeepGP.git#egg=PyDeepGP'
      ]
      )
