# Cross Domain Object Detection

This is a project page for a paper, Cross-Domain Weakly-Supervised Object Detection through Progressive Domain Adaptation [Inoue+, CVPR2018].

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

### Training original models
```
python train_model.py --root datasets/clipart --subset train --result result --det_type ssd300 --data_type clipart --gpu 0
```

## References
- [1]: N. Inoue et al. "Cross-Domain Weakly-Supervised Object Detection through Progressive Domain Adaptation", in CVPR, 2018.
