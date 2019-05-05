#Pytorch Implementation of Pixel-LINK

This repository is currently into active development. Do raise issues and we will solve them as soon as possible.

Public Board on Trello - https://trello.com/b/V26dOOOB

## A brief abstract of your project including the problem statement and solution approach

We are attempting to detect all kinds of text in the wild. The technique used for text detection is based on the paper PixelLink: Detecting Scene Text via Instance Segmentation (https://arxiv.org/abs/1801.01315) by Deng et al. The text instances present in the scene images lie very close to each other, and it is challenging to distinguish them using semantic segmentation. So, there is a need of instance segmentation. 

The approach consists of two key steps: 
a) Linking of pixels in the same text instance - Segmentation step, 
b) Text bounding box extraction using the linking done.

There are two kinds of predictions getting done here at each pixel level in the image: 
a) Text/non-text prediction, 
b) Link prediction.

This approach sets it apart from other kinds of methodologies used so far for text detection. Before PixelLink, the SOTA approaches on text detection does two kinds of prediction: a) Text/non-text prediction, b) Location Regression. Here both of these predictions are made at one go taking many fewer number of iterations and less training data.

## A list of code dependencies.

It is present in the file requirements.txt

## Detailed instructions for running the code, preferably, command instructions that may reproduce the declared results. If your code requires a model that can't be provided on GitHub, store it somewhere else and provide a download link.



## Results: If numerical, mention them in tabular format. If visual, display. If you've done a great project, this is the area to show it!

## Additional details, discussions, etc.


## References.
1. Deng, Dan, et al. "Pixellink: Detecting scene text via instance segmentation." Thirty-Second AAAI Conference on Artificial Intelligence. 2018.
2. Karatzas, Dimosthenis, et al. "ICDAR 2015 competition on robust reading." 2015 13th International Conference on Document Analysis and Recognition (ICDAR). IEEE, 2015.
3. VGG Synth Text in the wild: A. Gupta, A. Vedaldi, A. Zisserman "Synthetic Data for Text Localisation in Natural Images" IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
4. Ren, Mengye, and Richard S. Zemel. "End-to-end instance segmentation with recurrent attention." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.
5. Girshick, Ross. "Fast r-cnn." Proceedings of the IEEE international conference on computer vision. 2015.

