# Overview
We provide three datasets (Clipart1k, Watercolor2k, Comic2k).

Please refer to the paper for further details about the datasets.

**Note that these datasets are meant for education and research purposes only.**

We do not hold the copyright to the images in the datasets, but to avoid the tedium of downloading and processing the data, we are making available our local copy of the data.

# Details for the datasets
We followed [PASCAL VOC format](http://host.robots.ox.ac.uk/pascal/VOC/) basically.

The structure of the dataset is as follows:

- `JPEGImages`: images (like `<id>.jpg`)
- `ImageSets/Main`: list of ids for each subset (i.e., `train`/`test`/`extra`)
- `Annotations`: annotations (like `<id>.xml`)

We describe an xml element that indicates origin of an image in each dataset.

- `cmplaces_id`: When the image is collected from [CMplaces](http://projects.csail.mit.edu/cmplaces/).
- `src`: When the image is collected from image search engines.
- `bam_src`: When the image is collected from [BAM](https://bam-dataset.org/)

### dummy annotations
`extra` subset contains only image-level annotations.

We inserted dummy `bndbox` element although the category indicated by `name` is valid (actually existing in the image).

```
         <object>
                <name>person</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>1</xmin>
                        <ymin>1</ymin>
                        <xmax>1</xmax>
                        <ymax>1</ymax>
                </bndbox>
        </object>
```

# Setup

## Download original datasets

```
bash prepare.sh
```

## Download domain-transferred images for step1 (CycleGAN)
This script creates `dt_<domain>` directories.
```
bash prepare_dt.sh
```

## Prepare datasets for step2 (pseudo labeling after step1)
This script creates `dt_pl_<domain>` directories.
```
bash prepare_dt_pl.sh
```
Note that `Annotations` will be created by `pseudo_label.py`.
