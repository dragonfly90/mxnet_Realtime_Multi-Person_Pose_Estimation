import os
from os.path import join as pjoin
from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

def find_in_path(name, path):
    "Find a file in a search path"
    # Adapted fom
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

def customize_compiler_for_nvcc(self):
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

ext_modules = [
    Extension(
        "heatmap",
        ["heatmap.pyx"],
        extra_compile_args = {'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    ),
    Extension(
        "pafmap",
        ["pafmap.pyx"],
        extra_compile_args = {'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    ),
]

# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

setup(
    name = 'fpose_cython',
    ext_modules = ext_modules,
    # inject our custom trigger
    cmdclass = {'build_ext': custom_build_ext},
)
