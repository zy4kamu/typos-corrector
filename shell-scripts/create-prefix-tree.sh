#!/bin/bash

pushd ../build
./prepare-dataset-app -c prefix-tree -t ../corpus --prefix-tree-file ../python/model/prefix-tree
popd
