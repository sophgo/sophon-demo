#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

package = Extension('bbox', ['box_overlaps.pyx'], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([package]))
