#!/bin/bash

pushd ../build
cmake ../cplusplus -DCMAKE_BUILD_TYPE=DEBUG
make
popd
