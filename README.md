Assignment: 

Check this Repo out: https://github.com/kuangliu/pytorch-cifar
(Optional) You are going to follow the same structure for your Code (as a reference). So Create:
models folder - this is where you'll add all of your future models. Copy resnet.py into this folder, this file should only have ResNet 18/34 models. Delete Bottleneck Class
main.py - from Google Colab, now onwards, this is the file that you'll import (along with the model). Your main file shall be able to take these params or you should be able to pull functions from it and then perform operations, like (including but not limited to):
training and test loops
data split between test and train
epochs
batch size
which optimizer to run
do we run a scheduler?
utils.py file (or a folder later on when it expands) - this is where you will add all of your utilities like:
image transforms,
gradcam,
misclassification code,
tensorboard related stuff
advanced training policies, etc
etc
Your assignment is to build the above training structure. Train ResNet18 on Cifar10 for 20 Epochs. The assignment must:
pull your Github code to google colab (don't copy-paste code)
prove that you are following the above structure
that the code in your google collab notebook is NOTHING.. barely anything. There should not be any function or class that you can define in your Google Colab Notebook. Everything must be imported from all of your other files
your colab file must:
train resnet18 for 20 epochs on the CIFAR10 dataset
show loss curves for test and train datasets
show a gallery of 10 misclassified images
show gradcamLinks to an external site. output on 10 misclassified images. Remember if you are applying GradCAM on a channel that is less than 5px, then please don't bother to submit the assignment. 😡🤬🤬🤬🤬
Once done, upload the code to GitHub, and share the code. This readme must link to the main repo so we can read your file structure. 
Train for 20 epochs
Get 10 misclassified images
Get 10 GradCam outputs on any misclassified images (remember that you MUST use the library we discussed in the class)
Apply these transforms while training:
RandomCrop(32, padding=4)
CutOut(16x16)
Assignment Submission Questions:

Share the COMPLETE code of your model.py or the link for it
Share the COMPLETE code of your utils.py or the link for it
Share the COMPLETE code of your main.py or the link for it
Copy-paste the training log (cannot be ugly)
Copy-paste the 10/20 Misclassified Images Gallery
Copy-paste the 10/20 GradCam outputs Gallery
Share the link to your MAIN repo
Share the link to your README of Assignment  (cannot be in the MAIN Repo, but Assignment 11 repo)



</br>**References**</br>
</br>https://github.com/kuangliu/pytorch-cifar</br>
</br>https://github.com/jacobgil/pytorch-grad-cam</br>
