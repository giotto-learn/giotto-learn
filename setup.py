#! /usr/bin/env python
"""Toolbox for Machine Learning using Topological Data Analysis."""

import os
import codecs
import re
import sys
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


version_file = os.path.join('gtda', '_version.py')
with open(version_file) as f:
    exec(f.read())

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

DISTNAME = 'giotto-tda'
DESCRIPTION = 'Toolbox for Machine Learning using Topological Data Analysis.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_TYPE = 'text/x-rst'
MAINTAINER = 'Umberto Lupo, Lewis Tunstall'
MAINTAINER_EMAIL = 'maintainers@giotto.ai'
URL = 'https://github.com/giotto-ai/giotto-tda'
LICENSE = 'GNU AGPLv3'
DOWNLOAD_URL = 'https://github.com/giotto-ai/giotto-tda/tarball/v0.2.0'
VERSION = __version__ # noqa
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: C++',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8']
KEYWORDS = 'machine learning, topological data analysis, persistent ' + \
    'homology, persistence diagrams, Mapper'
INSTALL_REQUIRES = requirements
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov',
        'pytest-azurepipelines',
        'pytest-benchmark',
        'jupyter_contrib_nbextensions',
        'flake8',
        'hypothesis'],
    'doc': [
        'openml',
        'sphinx',
        'nbconvert',
        'sphinx-issues',
        'sphinx_rtd_theme',
        'numpydoc'],
    'examples': [
        'jupyter',
        'pandas',
        'openml']
}


def combine_requirements(base_keys):
    return list(
        set(k for v in base_keys for k in EXTRAS_REQUIRE[v]))


EXTRAS_REQUIRE["dev"] = combine_requirements(
    [k for k in EXTRAS_REQUIRE if k != "examples"])


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


url = "https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.zip"
class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the "
                               " following extensions: " +
                               " , ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                                   out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

            # TO DELETE
            import os
            from pathlib import Path
            import urllib.request
            import shutil
            import zipfile

            boost_folder = "C:\local"

            Path(boost_folder).mkdir(parents=True, exist_ok=True)
            file_name = os.path.join(boost_folder, "1_72_0")
            with urllib.request.urlopen(url) as response, \
                open(file_name+".zip", 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

            with zipfile.ZipFile(file_name, 'r') as zip_ref:
                zip_ref.extractall(file_name)

        self.install_dependencies()

        for ext in self.extensions:
            self.build_extension(ext)

    def install_dependencies(self):
        dir_start = os.getcwd()
        dir_pybind11 = os.path.join(dir_start,
                                    'gtda', 'externals', 'pybind11')
        if os.path.exists(dir_pybind11):
            return 0
        os.mkdir(dir_pybind11)
        subprocess.check_call(['git', 'clone',
                               'https://github.com/pybind/pybind11.git',
                               dir_pybind11])

        subprocess.check_call(['git', 'submodule', 'update',
                               '--init', '--recursive'])

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.join(os.path.dirname(
            self.get_ext_fullpath(ext.name)), 'gtda', 'externals', 'modules'))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == 'Windows':
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}'
                           f'={extdir}']
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)


setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESCRIPTION_TYPE,
      zip_safe=False,
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      keywords=KEYWORDS,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      ext_modules=[CMakeExtension('gtda')],
      cmdclass=dict(build_ext=CMakeBuild))
