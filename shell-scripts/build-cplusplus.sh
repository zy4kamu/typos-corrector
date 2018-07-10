#!/bin/bash

if [ ! -d ../build ]; then
    mkdir ../build
fi

pushd ../build
cmake ../cplusplus -DCMAKE_BUILD_TYPE=RELEASE
make
popd
