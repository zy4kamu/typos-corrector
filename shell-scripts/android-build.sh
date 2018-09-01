#!/bin/bash

android_env_folder=/tmp/android-env-folder
device_application_folder=/data/local/tmp/typos-corrector
application_folder=../android-application
build_folder=$android_env_folder/build
input_folder=../cplusplus
src_folder=$android_env_folder/typos-corrector

# create virtual environment for android build
if [ ! -d $android_env_folder ]; then
    make_standalone_toolchain.py --arch arm --stl=libc++ --install-dir $android_env_folder
fi

# definitions for android build
export LD_LIBRARY_PATH=""
export CPLUS_INCLUDE_DIRECTORY=""
export CC="$android_env_folder/bin/clang"
export CXX="$android_env_folder/bin/clang++"

# build project
rsync -avz $input_folder/ $src_folder/
if [ ! -d $build_folder ]; then
    mkdir $build_folder
fi
pushd $build_folder
cmake $src_folder -DCMAKE_BUILD_TYPE=RELEASE -DPLATFORM=ANDROID -DBUILD_PYTHON_BINDINGS=OFF
make
popd

# copy application to the output folder
rm -rf $application_folder
mkdir $application_folder
cp $build_folder/network-hypo-searcher/network-hypo-searcher $application_folder
#cp -R ../python/model/dataset $application_folder
cp -R ../python/model/parameters $application_folder
cp -R ../python/model/first-mistake-statistics $application_folder

# copy application to device and connect there
#adb shell rm -r $device_application_folder
adb shell mkdir -p $device_application_folder
adb push $application_folder $device_application_folder
