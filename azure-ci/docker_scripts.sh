#!/bin/bash

set -e -x

echo 'folders: '
ls
# update-alternatives --set python /opt/python/cp37-cp37m/
/opt/python/*$PYTHON_VER*/bin/pip install --upgrade pip setuptools
which cmake
for PYBIN in /opt/python/*/bin; do "${PYBIN}/pip" install cmake; done
ls /opt/python/cp37-cp37m/bin/
CMAKE_BIN=/opt/python/cp37-cp37m/bin/cmake
ln -sf ${CMAKE_BIN} /usr/bin/cmake

# install boost
# /usr/bin/wget / https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz
# tar -zxvf /boost_1_69_0.tar.gz

/opt/python/cp37-cp37m/bin/pip install -e "/io/.[tests, doc]"
/opt/python/cp37-cp37m/bin/pip uninstall -y giotto-learn
/opt/python/cp37-cp37m/bin/pip install wheel twine

pytest --cov /io/giotto/ --cov-report xml
flake8 --exit-zero /io/
displayName: 'Test with pytest and flake8'

python /io/setup.py sdist bdist_wheel
