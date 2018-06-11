# Cross-Domain Weakly-Supervised Object Detection through Progressive Domain Adaptation 

This page is for a paper which is to appear in CVPR2018 [1].
You can also find project page for the paper in [2].

Here is the example of our results in watercolor images.
![fig](dets_watercolor.png)

## Requirements
- Python 3.5+
- Chainer 3.0+
- ChainerCV 0.8+
- Cupy 2.0+
- Matplotlib

## Download models
Please go to both `models` and `datasets` directory and follow the instructions.

## Usage
For more details about arguments, please refer to `-h` option or the actual codes.

### Demo using trained models
```
python demo.py input/watercolor_142090457.jpg output.jpg --gpu 0 --load models/watercolor_ssd300
```

### Evaluation of trained models
```
python eval_model.py --root datasets/clipart --data_type clipart --det_type ssd300 --gpu 0 --load models/clipart_ssd300
```

### Training using clean instance-level annotations (ideal case)
```
python train_model.py --root datasets/clipart --subset train --result result --det_type ssd300 --data_type clipart --gpu 0
```

### Training using virtually created instance-level annotations

Work in progress..

### Citation

If you find this code useful for your research, please cite our paper:

```
@InProceedings{Inoue_2018_CVPR,
author = {Inoue, Naoto and Furuta, Ryosuke and Yamasaki, Toshihiko and Aizawa, Kiyoharu},
title = {Cross-Domain Weakly-Supervised Object Detection Through Progressive Domain Adaptation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```

## References
- [1]: [N. Inoue et al. "Cross-Domain Weakly-Supervised Object Detection through Progressive Domain Adaptation", in CVPR, 2018.](https://arxiv.org/abs/1803.11365)
- [2]: [Project page](https://naoto0804.github.io/cross_domain_detection/)
