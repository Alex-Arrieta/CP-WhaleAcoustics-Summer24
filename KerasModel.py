import tensorflow as tf
import argparse
import os
import json
import pandas as pd

#Much of this code is based on the AWS SageMaker Tensorflow SDK documentaiton
#https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html


if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.1)

    # an alternative way to load hyperparameters via SM_HPS environment variable.
    parser.add_argument('--sm-hps', type=json.loads, default=os.environ['SM_HPS'])

    # input data and model directories
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING']) #This is passed by the user
    parser.add_argument('--labels', type=str, default=os.environ['SM_CHANNEL_LABELS'])

    args = parser.parse_args()
    
    #This defines the acutal keras model.
    #A custom model implementation may be used if desired
    model = tf.keras.applications.ResNet152V2(
        weights=None,
        input_tensor=None,
        input_shape=(192,100,3),
        pooling=None,
        classes=1,
        classifier_activation=None
    )
    #This pull all the labels from the training manifest file provided under --labels
    #Note that this is a local directory for the file as all the data gets copied from S3 into this job
    training_labels = pd.read_csv(args.labels + "/train_lst.lst", sep = "\t", names = ["index", "Classification", "File"])
    training_labels = training_labels.sort_values(by=['File'])
    training_labels = training_labels["Classification"].to_numpy().tolist()
    #Reads all the training images
    x_train = tf.keras.preprocessing.image_dataset_from_directory(args.training, 
                                                                  labels = training_labels,
                                                                  image_size = (192,100), 
                                                                  label_mode = "binary", 
                                                                  color_mode = "rgb",
                                                                  batch_size = None) #Use "grayscale" if the input image was greyscale
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=['accuracy']
    )
    model.fit(x_train.map(lambda x, y: (tf.keras.applications.resnet_v2.preprocess_input(x), y)).batch(args.batch_size), epochs = args.epochs)
    #This saves the model. The file name needs to be a bunch of numbers for some reason
    #Careful using the .keras save format as the version of tensor flow a SageMaker Training job can use is different than the one on notebooks (The notebooks have a newer version)
    model.save(os.path.join(os.environ['SM_MODEL_DIR'], '000001'), save_format='tf')