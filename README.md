# DECISION
Unsupervised Multi-source Domain Adaptation Without Access to Source Data (CVPR '21 Oral)

### Overview
This repository is a PyTorch implementation of the paper [Unsupervised Multi-source Domain Adaptation Without Access to Source Data](https://arxiv.org/pdf/2104.01845.pdf) published at [CVPR 2021](http://cvpr2021.thecvf.com/). (alpha build)

### Dataset
- Manually download the datasets [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [Office-Caltech](http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar) from the official websites.
- Generate '.txt' file for each dataset using `gen_list.py` (change dataset argument in the file accordingly). 

### Training
- Train source models (shown here for Office with source A)
```
python image_source.py --dset office --s 0 --max_epoch 100 --trte val --gpu_id 0 --output ckps/source/
```
- Adapt to target (shown here for Office with target D)
```
python adapt_multi.py --dset office --t 1 --max_epoch 100 --gpu_id 0 --output_src ckps/source/ --output ckps/adapt
```
