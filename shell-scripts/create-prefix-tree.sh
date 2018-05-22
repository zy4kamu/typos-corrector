#!/bin/bash

pushd ../build
./prepare-dataset -c prefix-tree -t ../corpus --prefix-tree-file ../python/model/prefix-tree
popd
