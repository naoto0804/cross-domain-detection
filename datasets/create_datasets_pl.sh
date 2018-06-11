#!/usr/bin/env bash

root='.'
names=(clipart watercolor comic)
for name in "${names[@]}"
do
    mkdir pl_${name}
    mkdir pl_${name}/Annotations
    ln -s ${root}/${name}/ImageSets pl_${name}/ImageSets
    ln -s ${root}/${name}/JPEGImages pl_${name}/JPEGImages
done
