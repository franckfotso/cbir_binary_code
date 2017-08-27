# cbir_binary_code

Content Based Image Retrieval with Binary Hash Code

Created by Romyny

## Introduction:

CBIR, which stands for Content Based Image Retrieval, relies on extraction of features from images and then comparing the images for similarity based on the extracted feature vectors and distance metric. However, the CBIR system also involves methods to:
  * Efficiently store features extracted of images
  * Scale the time of request for a dataset with a high size
  * Combine techniques from computer vision, information retrieval, and powerful data structure to build real-world images search engines that can be deployed online

Since there were many researchs related to CBIR using basics techniques of computer vision, we proprosed a repository which focus on how deep can we involved Deep Learning techniques in CBIR. So, our work is about the use of **binary hash code** (proposed by [**Kevin Lin - cvprw15**](https://github.com/kevinlin311tw/caffe-cvprw15)) for CBIR tasks.

## Purposes:

Our main goals are:
  * Propose a tool for the extraction of binary hash codes & deep features
  * Fast indexing of both binary hash codes & deepfeatures
  * Fast computing of similarity (distances) based on features & binary codes extrated
  * Easy request for similar images (in database)
  * Sorted visualization of results

## Datasets:

We have first tested our tool on CIFAR-10, then we have built our own datasets (foods & electronical products). As a result, we were able to see our tool' behaviour on the real-world issues. Please have a look on **INSTALLATION** section to download our datasets.

## Hardwares/Softwares:
    OS: Ubuntu 16.04 64 bit
    GPU: Nvidia GTX 950M
    Cuda 8.0
    CuDNN 4.0.7
    Python 2.7.12
    OpenCV 3.1.0
    
## Prerequisites:

  1. Caffe's [prerequisites](http://caffe.berkeleyvision.org/installation.html#prequequisites)
  2. Python's packages (requirements.txt)

## Installation:

To install this tool, please follow the steps below:

1. [Install OpenCV](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)

2. Download the repository:

    ```
    $ cd /opt
    $ sudo git clone https://github.com/romyny/cbir_binary_code.git
    $ cd cbir_binary_code
    ```
  
3. Install caffe in 'cbir_binary_code/caffe':

    Adjust Makefile.config, then
    
    ```
    $ cd caffe
    $ sudo mkdir build
    $ cd build
    $ cmake ..
    $ make -j8
    $ make install
    ```
  
4. Install python's packages required:

  ```
  for req in $(cat requirements.txt); do pip install $req; done
  ```
  
Get the data and models required:
1. Download the data and uncompress in '/opt/cbir_binary_code/data'
  * data_foods25: [Google Drive](https://drive.google.com/open?id=0B_Rjj_NgCayPRExDYkNKTWF1bjQ)
  * data_product20: [Google Drive](https://drive.google.com/open?id=0B_Rjj_NgCayPcEVqTW9wTE1tRjg)
  
2. Download the models and uncompress in '/opt/cbir_binary_code/examples'
  * example_foods25: [Google Drive](https://drive.google.com/open?id=0B_Rjj_NgCayPcC1kNXlRWmRWY2M)
  * example_product20: [Google Drive](https://drive.google.com/open?id=0B_Rjj_NgCayPYjRBRUtPcG5MeXM)

## Experiments:

1. Feature extraction & indexing of a sample dataset (products20) with the following commands:
    
    ```
    $ cd /opt/cbir_binary_code
    $ bash tools/indexing.sh 1 \
    /opt/cbir_binary_code/examples/foods25/RomynyNet_foods25_48_iter_30000.caffemodel \
    /opt/cbir_binary_code/examples/main/deploy_fc7.prototxt \
    /opt/cbir_binary_code/examples/foods25/RomynyNet_foods25_48_deploy.prototxt \
    /opt/cbir_binary_code/data/foods25/imgs \
    /opt/cbir_binary_code/data/foods25/foods25_48_deepDB.hdf5 foods25
    ```
  
  The output of this command is stored as **'/opt/cbir_binary_code/data/foods25/foods25_48_deepDB.hdf5'**.
  The log files are stored in **'/opt/cbir_binary_code/logs'**.
  
2. Search for similar images:

    ```
    $ cd /opt/cbir_binary_code
    $ python tools/deep_search.py --use_gpu 1 \
    --model_file /opt/cbir_binary_code/examples/products20/RomynyNet_products20_48_iter_20000.caffemodel \
    --feat_proto /opt/cbir_binary_code/examples/main/deploy_fc7.prototxt \
    --bin_proto /opt/cbir_binary_code/examples/products20/RomynyNet_products20_48_deploy.prototxt \
    --products_dir /opt/cbir_binary_code/data/products20/imgs \
    --deep_db /opt/cbir_binary_code/data/products20/products20_48_deepDB.hdf5 \
    --query /opt/cbir_binary_code/data/products20/imgs/moulinex/07072053.jpg
    ```
  
  The output of this command is stored in **'/opt/cbir_binary_code/output/results/'**

## Our results

1. Query on product_1:
![GitHub Logo](/demo/00473862.png)

2. Query on product_2:
![GitHub Logo](/demo/14904540.png)

3. Query on product_3:
![GitHub Logo](/demo/21787018.png)


## Contact

Please feel free to leave suggestions or comments to Romyny (romyny9096@gmail.com)
