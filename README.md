## Saliency Prediction System

### An Encoder-Decoder Architecture using VGG16 as a base network to predict eye fixation maps

Download dataset from [here](https://drive.google.com/file/d/1zL3mZ4Qv8XWzHEIbnMwyJ7--PYZzIfSD/view?usp=sharing)

1. If training on Google Colab then upload the dataset and mount the drive.
2. If training on personal GPU, comment the code for google drive mounting and change the paths for train and validation datasets.

3. While test.py doesn't need much GPU and can also be done on CPU 

   * Before executing test.py provide paths for the main directory and for test-images directory. 
   
   * Also check if the model is loading properly or not.
