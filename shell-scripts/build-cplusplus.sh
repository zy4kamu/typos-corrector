#!/bin/bash

src_folder='../cplusplus'
build_folder='../build'
application_folder='../application'

if [ ! -d $build_folder ]; then
    mkdir $build_folder
fi

pushd $build_folder
cmake $src_folder -DCMAKE_BUILD_TYPE=RELEASE -DUSE_OPENCL=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=1
rc -J .
make
popd

#rm -rf $application_folder
#mkdir $application_folder
#cp $build_folder/network-hypo-searcher/network-hypo-searcher $application_folder
#cp -R ../python/model/dataset $application_folder
#cp -R ../python/model/parameters $application_folder
#cp -R ../python/model/first-mistake-statistics $application_folder
