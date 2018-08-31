#!/bin/bash

export LD_LIBRARY_PATH=""
export CPLUS_INCLUDE_DIRECTORY=""

ANDROID_FOLDER=/tmp/android-dev-folder
#PROJECT_FOLDER=../cplusplus
PROJECT_FOLDER=/home/stepan/git-repos/clBLAS

rm -rf ${ANDROID_FOLDER}
make_standalone_toolchain.py --arch arm --stl=libc++ --install-dir ${ANDROID_FOLDER}
export CC="${ANDROID_FOLDER}/bin/clang"
export CXX="${ANDROID_FOLDER}/bin/clang++"

cp -R ${PROJECT_FOLDER} ${ANDROID_FOLDER}/cplusplus
mkdir ${ANDROID_FOLDER}/build
pushd ${ANDROID_FOLDER}/build
cmake ../cplusplus/src -DCMAKE_BUILD_TYPE=RELEASE -DPLATFORM=ANDROID
make
popd
