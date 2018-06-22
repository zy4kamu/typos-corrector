#!/bin/bash

export PYTHONPATH=/opt/anaconda2/lib/python2.7/site-packages

pushd ../python
/opt/anaconda2/bin/python demo_server.py
popd
