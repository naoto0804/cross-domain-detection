#!/usr/bin/env bash

root='.'
names=(clipart watercolor comic)
for name in "${names[@]}"
do
    mkdir dt_pl_${name}
    mkdir dt_pl_${name}/Annotations
    cp -r ${root}/${name}/ImageSets dt_pl_${name}/ImageSets
    ln -s ${root}/${name}/JPEGImages dt_pl_${name}/JPEGImages
done
