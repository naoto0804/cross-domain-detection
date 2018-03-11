#!/usr/bin/env bash

names=(clipart watercolor comic)
for name in "${names[@]}"
do
    curl -O http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/cross_domain_detection/models/${name}_ssd300
done
