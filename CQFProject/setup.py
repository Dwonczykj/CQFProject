#!/usr/bin/env python

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'CQF Final Project Credit Spreads',
  version='1.0',
  author='Joseph Dwonczyk',
  author_email='joe.dwonczyk@riverandmercantile.com',
  packages=['Cython'],
  ext_modules = cythonize(["LowDiscrepancyNumberGenerators.pyx","SimulateLegs.pyx","Copulae.pyx","EmpiricalFunctions.pyx","CumulativeAverager.pyx"]),
  include_dirs=[numpy.get_include()]
)
