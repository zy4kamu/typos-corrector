#!/bin/bash

pushd ../build
./typos-corrector-helper-app -c prefix-tree -d ../python/model/dictionary --prefix-tree-file ../python/model/prefix-tree
popd
