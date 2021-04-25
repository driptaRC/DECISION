# DECISION
Unsupervised Multi-source Domain Adaptation Without Access to Source Data (CVPR '21 Oral)

### Overview
This repository is a PyTorch implementation of the paper [Unsupervised Multi-source Domain Adaptation Without Access to Source Data](https://arxiv.org/pdf/2104.01845.pdf) published at [CVPR 2021](http://cvpr2021.thecvf.com/). This code is based on the [SHOT](https://github.com/tim-learn/SHOT) repository.

### Dependencies
Create a conda environment with `environment.yml`.

### Dataset
- Manually download the datasets [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [Office-Caltech](http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar) from the official websites.
- Move `gen_list.py` inside data directory.
- Generate '.txt' file for each dataset using `gen_list.py` (change dataset argument in the file accordingly). 

### Training
- Train source models (shown here for Office with source A)
```
python train_source.py --dset office --s 0 --max_epoch 100 --trte val --gpu_id 0 --output ckps/source/
```
- Adapt to target (shown here for Office with target D)
```
python adapt_multi.py --dset office --t 1 --max_epoch 15 --gpu_id 0 --output_src ckps/source/ --output ckps/adapt
```
- Distill to single target model (shown here for Office with target D)
```
python distill.py --dset office --t 1 --max_epoch 15 --gpu_id 0 --output_src ckps/adapt --output ckps/dist
```

### Citation
If you use this code in your research please consider citing
```
@article{ahmed2021unsupervised,
  title={Unsupervised Multi-source Domain Adaptation Without Access to Source Data},
  author={Ahmed, Sk Miraj and Raychaudhuri, Dripta S and Paul, Sujoy and Oymak, Samet and Roy-Chowdhury, Amit K},
  journal={arXiv preprint arXiv:2104.01845},
  year={2021}
}
```
