Background Segmentation for Deep Style
===
This code implements the model discussed in Background Segmentation for Deep Style. The model is able to segement the background 'suff' such as 'sea' or 'grass' and style it with different textures and brush size. We also use the model for segmenting 'cars' on a road in autonomous driving. 

Requirements
---
This code is based on the pix2pix code found at:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

Preprocessing
---
The training data is in the form of triples : A (original image), B (styled background), C (segmentation mask)

![sample_input](https://user-images.githubusercontent.com/65399216/90971912-b46a4400-e557-11ea-9945-b5a6eb8eaa2e.jpg)

Training
---
Train the model:
python train.py --dataroot ./datasets/dataset_name --name model_instance_name --model sd --direction AtoB --dataset_mode triple
 - The training data will be taken from ./datasets/dataset_name/train
 - "--model sd" is used to set the training model as the one defined in sd_model.py
 - "--dataset_mode triple" is used to the dataloader as the one defined in triple_dataset.py
 - Training results found under ./checkpoint


Testing
---
python test.py --dataroot ./datasets/dataset_name --name model_instance_name --model sd --direction AtoB --num_test 300 --dataset_mode triple
 - The testing data will be taken from ./datasets/dataset_name/test
 - "--num_test" indicates the number of images to use from test set
 - Testing results found under ./results

Test Sample for Car Segmentation
---
Sample 1 For Input and Generated Segmentation for Car video :

![target1](https://user-images.githubusercontent.com/65399216/90971921-dd8ad480-e557-11ea-93b1-56cd58fc064c.gif)
![generated1](https://user-images.githubusercontent.com/65399216/90971923-e085c500-e557-11ea-8693-733c9eab41a6.gif)

Sample 2 For Input and Generated Segmentation for Car video :

![target2](https://user-images.githubusercontent.com/65399216/90971924-e380b580-e557-11ea-883c-02c47951d89e.gif)
![generated2](https://user-images.githubusercontent.com/65399216/90971925-e67ba600-e557-11ea-83ce-002604576d45.gif)


