
# About Dataset

This dataset available on [Kaggle](https://www.kaggle.com/shanks0465/braille-character-dataset) includes Braille script letter scans. The dataset is light (about 1 MB), it contains scans of the 26 characters of English in Braille script with augmentations as noted:

- dim (brightness)
- rot (rotation)
- whs (width height shift)

These augmentations had 20 variations each (with different intensities) for each alphabet in English language, in 28x28 pixels, jpeg format.
Thus, the total dataset included: 


**3 (augmentation) x 20 (augmentation intensity) x 26 (charecters) = 1,560 images**



This dataset was carefully picked for having 1000+ images and no R-code on kaggle available at the time of making the selection. Which makes it a perfect exercise for ML on portable computers.


Link to the [dataset page](https://www.kaggle.com/shanks0465/braille-character-dataset)


# Discussion and application

Today there are over 253 million visually impaired persons in the world, out of which 36 million are accounted for as completely blind persons. With the world moving towards AI and IoT, incorporating machine learning for various scripts/languages to decipher messages is not just a need for aiding specially abled but also a need for securirty tomorrow. Employing CNN for reading brail script in this assignment has proved to be effective. 
In future a similar machine learning alorithm can also be used to create programs for various applications, such as hand-writing recognition, deciphering ancient scripts, understanding damaged documents, intelligence generation, etc. 

# What you will find in Rmd

## Objective 1: Importing images and splitting test-train-validation sets
Places files into respective folder, prepares them for the ML training.

## Objective 2: Train model (Keras - CNN)
I used Keras - CNN model here for ML, the objective here is to be able to have the machine identify each charecter. Which is why we made 26 different folders for 26 letters with 3 augmentations each. Augmentations are to improve the readability for machine. I have used the augmentations provided in the dataset, but you can further augment these images to imrpove accuracy of the model. Augementation here means simply turning, rotating, changing light variance (dim) so that the computer can understand the difference between actual fearures to track and rest of the noise (or background) in the image.

## Objective 3: Plot relevant graphs
This plots the history curves, helps us understand how well the model performed.
