#!/bin/bash

pushd ../build
./typos-corrector ../python/model/dictionary ../python/model/prefix-tree
popd
