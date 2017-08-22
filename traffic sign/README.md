# **Traffic Sign Recognition Classifier** 

# *Submissions*
Traffic_Sign_Classifier.ipynb : Build a Traffic Signs Classifier. Get 93.4.0% accuracy on the test dataset using one Convolutional neural network. The trained model only get 40% accuracy on 10 new images from web.
report.html :  html verson of Traffic_Sign_Classifier.ipynb.
test fold: 10 new images from web.


### Reflection

### 1. Describe of pipeline.
1. Load Dataset;
2. Give distribution of classes in the training, validation and test set; Visulize an random image in train data set;
3. Preprocess train dataset using augmentation, center normalize, RGB2gray techniques;
4. Define Model Architecture,train pipeline, model evaluate funtion;
5. Train model using L2 normalization pennalty and dropout techniques to prevent overfitting;
6. Use test dataset to evaluate trained model.
7. Test trained model on new images(use same preprocess pipeline above) and get a 40% accuracy (not good);
8. Output top 5 softmax probabilities for each new test image .

### 2. shortcomings
1. Trained model only get 97% accuracy more or less on valid dataset.
2. Center normalization seems not useful for improving validating accuracy.

### 3. Suggest possible improvements to project
1. Maybe there is some other data augmentation techniques that may help.
2. Through analysis classificaton bad cases to get some helpful information on model accuracy promoting.


