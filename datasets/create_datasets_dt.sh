#!/usr/bin/env bash

root='.'
names=(clipart watercolor comic)
for name in "${names[@]}"
do
    mkdir dt_${name}
    mkdir dt_${name}/JPEGImages
    ln -s ${root}/${name}/ImageSets dt_${name}/ImageSets
    ln -s ${root}/${name}/Annotations dt_${name}/Annotations
done
