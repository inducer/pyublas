#! /bin/sh

g++ -shared \
  -I/usr/include/python2.5 \
  -I$HOME/pool/include/boost-1_35 \
  -I$HOME/pool/include/boost-bindings \
  -L$HOME/pool/lib \
  -lboost_python-gcc42-mt \
  test_ext.cpp \
  -o test_ext.so
