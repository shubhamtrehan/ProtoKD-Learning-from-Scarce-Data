# -*- coding: utf-8 -*-
"""ProtoKD_Shubham.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PSS8T9wO61_JF0SA22R9wd84sXKY_xny
"""

import sys

import tensorflow as tf
import multiprocessing as mp
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD,Adam
import pickle

from models import make_feat_extractor
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

if gpus:
  try:
    # Specify the GPU index here
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

"""<span style="font-size: 24px; color: yellow;"><b>Read Ova Data</b></span>"""

proto_samples_per_class = 20

# Define paths to your train and test directories
root = f"/media/DiskDrive1/Datasets/Ova_Dataset/Attempt4_data_version/{proto_samples_per_class}_Samples/ProtoNet_ProtoKD_data_version"
train_directory = f'{root}/train'
test_directory = f'{root}/test'

assert len(os.listdir(train_directory)) == len(os.listdir(test_directory))
all_classes = sorted(os.listdir(train_directory), key=str.lower)
all_classes.remove("Background")
print(all_classes)
print()
print("Total Number of Classes in this dataset",len(all_classes))

def read_category(category_directory_path, category_name):
    """
    Reads all images from a given category directory,
    applies rotations, and constructs labels including the
    category and rotation angle.
    """
    datax = []
    datay = []
    images = os.listdir(category_directory_path)
    for img in images:
        image_path = os.path.join(category_directory_path, img)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize image to 28x28 pixels
        image = cv2.resize(image, (128, 128))
        # Rotations of image
        rotated_90 = ndimage.rotate(image, 90, reshape=False)
        rotated_180 = ndimage.rotate(image, 180, reshape=False)
        rotated_270 = ndimage.rotate(image, 270, reshape=False)
        # Collect images
        datax.extend((image, rotated_90, rotated_180, rotated_270))
        # Construct labels
        img_name = os.path.splitext(img)[0]  # Remove file extension
        datay.extend((
            category_name, #+ '_' + img_name + '_0',
            category_name, # + '_' + img_name + '_90',
            category_name, # + '_' + img_name + '_180',
            category_name, # + '_' + img_name + '_270'
        ))
    return np.array(datax), np.array(datay)

def read_images(base_directory):
    """
    Reads all categories from the base_directory
    Uses multiprocessing to decrease the reading time
    """
    datax_list = []
    datay_list = []
    categories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    if "Background" in categories:
        categories.remove("Background")
    print("Number of classes",len(categories))
    pool = mp.Pool(mp.cpu_count())
    results = []
    for category in categories:
        category_directory_path = os.path.join(base_directory, category)
        result = pool.apply_async(read_category, args=(category_directory_path, category))
        results.append(result)
    pool.close()
    pool.join()
    for result in results:
        category_datax, category_datay = result.get()
        datax_list.append(category_datax)
        datay_list.append(category_datay)
    datax = np.vstack(datax_list)
    datay = np.concatenate(datay_list)
    return datax, datay



# Read images and labels from train and test directories
trainx, trainy = read_images(train_directory)
testx, testy = read_images(test_directory)

print(f"Train data shape: {trainx.shape}, Train labels shape: {trainy.shape}")
print(f"Test data shape: {testx.shape}, Test labels shape: {testy.shape}")

def visualize_samples(datax, datay, dataset_name):
    """
    Visualizes 2 images from each class in the dataset along with their labels.

    Parameters:
    - datax: numpy array of images (shape: [num_samples, height, width, channels])
    - datay: numpy array of labels corresponding to the images
    - dataset_name: string, name of the dataset (e.g., "Training Set")

    """
    unique_labels = np.unique(datay)
    unique_labels = unique_labels[unique_labels != "Background"]
    print(unique_labels)
    num_classes = len(unique_labels)
    print("Number of classes in this dataset are", len(unique_labels))
    samples_per_class = proto_samples_per_class  # Number of images to display per class

    # Calculate the number of rows and columns for the subplot grid
    total_plots = num_classes * samples_per_class
    cols = samples_per_class
    rows = num_classes

    plt.figure(figsize=(samples_per_class * 3, num_classes * 3))

    for i, label in enumerate(unique_labels):
        # Get indices of images with the current label
        indices = np.where(datay == label)[0]
        # Select up to 'samples_per_class' indices
        selected_indices = indices[:samples_per_class]
        for j, idx in enumerate(selected_indices):
            img = datax[idx]
            # Rescale image if pixel values are in [0, 1]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            # Determine subplot position
            plt_idx = i * samples_per_class + j + 1
            plt.subplot(rows, cols, plt_idx)
            # Handle grayscale images
            if img.ndim == 2 or img.shape[-1] == 1:
                plt.imshow(img.squeeze(), cmap='gray')
            else:
                plt.imshow(img)
            plt.axis('off')
            plt.title(f"Label: {label}", fontsize=8)
    plt.suptitle(f"{dataset_name} Samples", fontsize=80)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Visualize samples from the training set
visualize_samples(trainx, trainy, "Training Set")

# Visualize samples from the test set
# visualize_samples(testx, testy, "Test Set")

def extract_sample(n_way, n_support, n_query, datax, datay):
    """
    Picks random sample of size n_support+n_querry, for n_way classes
    Args:
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      datax (np.array): dataset of images
      datay (np.array): dataset of labels
    Returns:
      (dict) of:
        (torch.Tensor): sample of images. Size (n_way, n_support+n_query, (dim))
        (int): n_way
        (int): n_support
        (int): n_query
    """
    sample = []
    K = np.random.choice(np.unique(datay), n_way, replace=False)
    for cls in K:
        datax_cls = datax[datay == cls]
        perm = np.random.permutation(datax_cls)
        sample_cls = perm[:(n_support+n_query)]
        sample.append(sample_cls)

    sample = np.array(sample)
    sample = tf.convert_to_tensor(sample, dtype=tf.float32)

    return({
          'images': sample,
          'n_way': n_way,
          'n_support': n_support,
          'n_query': n_query
          })

def display_images_grid(sample):
    """
    Displays all images from the sample in a grid classwise
    Args:
        sample (tf.Tensor): Tensor of images with shape [8, 10, 3, 32, 32]
    """
    # Ensure the sample is a TensorFlow tensor
    sample = tf.convert_to_tensor(sample)

    # Determine the number of classes and images per class
    num_classes = sample.shape[0]
    images_per_class = sample.shape[1]

    # Create a figure to display images in a grid
    fig, axs = plt.subplots(num_classes, images_per_class, figsize=(images_per_class, num_classes))

    # Display images classwise
    for i in range(num_classes):
        for j in range(images_per_class):
            image = sample[i, j]
            image = tf.transpose(image, perm=[0, 1, 2]).numpy().astype(np.uint8)
            axs[i, j].imshow(image)
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage with TensorFlow tensors
# Assuming sample_example['images'] is a TensorFlow tensor with shape [8, 10, 3, 32, 32]
# sample_example = extract_sample(14, 5, 5, trainx, trainy)
# imags = tf.convert_to_tensor(sample_example['images'])
# display_images_grid(imags)

n_channels = 3
lr1 = 1e-4
depth=28
width=2
dropout_rate=0.3
img_size = (128, 128)

model = make_feat_extractor(input_shape=(img_size[0], img_size[1], n_channels),
                                              num_classes=len(all_classes),
                                              depth=depth, width=width, top = True,
                                              dropout_rate=dropout_rate, model_name='wide-resnet-ova')


model.summary()

def model_checking(inp_model):
    # Get the last layer from the top-level model
    last_layer = inp_model.layers[-1]

    # If the last layer is a nested Model, use its own last layer
    if isinstance(last_layer, tf.keras.Model):
        inner_last = last_layer.layers[-1]
        print("Last Layer Name:", inner_last.name)
        try:
            print("Activation Function:", inner_last.activation)
        except AttributeError:
            print("The inner last layer has no 'activation' attribute.")
    else:
        print("Last Layer Name:", last_layer.name)
        try:
            print("Activation Function:", last_layer.activation)
        except AttributeError:
            print("The last layer has no 'activation' attribute.")

    # Generate a random input tensor with shape (1, 128, 128, 3)
    random_input = np.random.random((1, 128, 128, 3)).astype(np.float32)

    # Get the model's output
    output = inp_model(random_input)
    # If the output is a list, choose the output with probabilities.
    if isinstance(output, (list, tuple)):
        output_tensor = output[-1]
    else:
        output_tensor = output

    # Convert the output tensor to a NumPy array
    output_array = output_tensor.numpy()

    # Sum the probabilities in the output
    probability_sum = np.sum(output_array)
    print(f"Sum of output probabilities: {probability_sum}")

# Run the function with your model
model_checking(inp_model=model)

def euclidean_distance(a, b):
    """
    Computes euclidean distance btw x and y
    Args:
      a (Query Tensor): shape (N, D). N usually n_way*n_query
      b (Proto Tensor): shape (M, D). M usually n_way
    Returns:
      Tensor: shape(N, M). For each query, the distances to each centroid
    """
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)

def calculate_loss(log_p_y, n_way, n_supp):
    """
    Calculates the classification loss from log probabilities.

    This function computes the loss by gathering the log probabilities corresponding to the
    correct class for each support example and then averaging them.

    Args:
        log_p_y (tf.Tensor): Tensor of log probabilities.
            Shape: (n_way, n_support, n_way).
        n_way (int): The number of classes.
        n_supp (int): The number of support examples per class.

    Returns:
        tf.Tensor: The mean loss.
    """

    temp = tf.Variable([])
    for i in range(0,n_way):
        for j in range(0,n_supp):
            temp = tf.experimental.numpy.append(temp, tf.gather_nd(log_p_y,indices=[[i,j,i]]))

    temp = tf.stack(temp, axis=0)
    return tf.reduce_mean(temp)

def calculate_acc(a_max_mat):
    """
    Calculates the classification accuracy.

    This function computes the accuracy by comparing the predicted class labels (argmax)
    with the true class labels.

    Args:
        a_max_mat (tf.Tensor): Tensor of predicted class labels (argmax).
            Shape: (n_way, n_support).

    Returns:
        tf.Tensor: The mean accuracy.
    """
    n_way = a_max_mat.shape[0]        #Number of Classes
    n_support = a_max_mat.shape[1]    #Number of Supporting samples
    row = [1 for t in range(n_support)]
    temp = tf.convert_to_tensor(np.array([np.array(row)*i for i in range(n_way)]))
    bool_mat = tf.cast(tf.math.equal(temp,a_max_mat), dtype=tf.float32)

    return tf.reduce_mean(bool_mat)

def get_support_query_from_extracted_sample(extracted_sample):
        """
        Extracts support and query sets from an extracted sample dictionary.

        This function takes an extracted sample dictionary containing images, n_way, n_support,
        and n_query information, and returns the support and query sets as TensorFlow tensors.

        Args:
            extracted_sample (dict): A dictionary containing the following keys:
                - 'images' (tf.Tensor): Tensor containing images for both support and query sets.
                    Shape: (n_way, n_support + n_query, height, width, channels).
                - 'n_way' (int): The number of classes.
                - 'n_support' (int): The number of support examples per class.
                - 'n_query' (int): The number of query examples per class.

        Returns:
            tuple: A tuple containing the support and query sets as TensorFlow tensors.
                - x_support (tf.Tensor): Tensor representing the support set.
                    Shape: (n_way * n_support, height, width, channels).
                - x_query (tf.Tensor): Tensor representing the query set.
                    Shape: (n_way * n_query, height, width, channels).
        """
        extracted_sample_images = extracted_sample['images']
        n_way = extracted_sample['n_way']
        n_support = extracted_sample['n_support']
        n_query = extracted_sample['n_query']


        x_support = extracted_sample_images[:, :n_support]

        x_query = extracted_sample_images[:, n_support:]

        x_support = tf.reshape(x_support, (x_support.shape[0]*x_support.shape[1], x_support.shape[2],
                                           x_support.shape[3], x_support.shape[4]))

        x_query = tf.reshape(x_query, (x_query.shape[0]*x_query.shape[1], x_query.shape[2],
                                           x_query.shape[3], x_query.shape[4]))

        return x_support, x_query

class Prototypical_Shubham(Model):
    def __init__(self, model):
        """
        Args:
            encoder : CNN encoding the images in sample
            n_way (int): number of classes in a classification task
            n_support (int): number of labeled examples per class in the support set
            n_query (int): number of labeled examples per class in the query set
        """
        super(Prototypical_Shubham, self).__init__()
        self.model = model

    def set_forward_loss(self, x_support, x_query):
        """
        Computes loss, accuracy and output for classification task
        Args:
            sample (Tensor): shape (n_way, n_support+n_query, (dim))
        Returns:
            Tensor: shape(2), loss, accuracy and y_hat
        """

        z_support = self.model(x_support)[0]

        z_query = self.model(x_query)[0]

        z_dim = z_support.shape[-1]

        z_support = tf.reshape(z_support, shape=[n_way,n_support,z_dim])

        z_proto = tf.reduce_mean(z_support, axis=1)

        dists = euclidean_distance(z_query, z_proto)

        log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [n_way, n_query, -1])

        loss_val =  calculate_loss(-log_p_y, n_way, n_support)

        y_hat = tf.argmax(log_p_y, axis=-1)    #Predictions

        acc_val = calculate_acc(y_hat)

        return loss_val, acc_val, #{'loss': loss_val,'acc': acc_val, 'y_hat': y_hat}

    def predPseudo(self, support, query):
        """
        Predicts pseudo-labels for query samples and calculates metric learning loss.

        This function takes support and query sets, concatenates them, encodes them using a model,
        calculates prototypes from the support set, and computes distances between query embeddings
        and prototypes. It also calculates a metric learning loss based on the similarity between
        query and prototype embeddings.

        Args:
            support (tf.Tensor): Tensor representing the support set.
                Shape: (n_way, n_support, height, width, channels), where:
                    - n_way is the number of classes.
                    - n_support is the number of support examples per class.
                    - height, width, channels are the dimensions of the input images.
            query (tf.Tensor): Tensor representing the query set.
                Shape: (n_way, n_query, height, width, channels), where:
                    - n_way is the number of classes.
                    - n_query is the number of query examples per class.
                    - height, width, channels are the dimensions of the input images.

        Returns:
            tuple: A tuple containing the distance matrix, query logits, and metric learning loss.
                - dists (tf.Tensor): Distance matrix between query embeddings and prototypes.
                    Shape: (n_way * n_query, n_way).
                - qry_logits (tf.Tensor): Logits for the query samples.
                    Shape: (n_way * n_query, num_classes).
                - loss_metric (tf.Tensor): Metric learning loss.
                """
        n_class = n_way
        cat = tf.concat([
            tf.reshape(support, [n_class * n_support,
                                 128, 128, 3]),
            tf.reshape(query, [n_class * n_query,
                               128, 128, 3])], axis=0)
        z, z_logits = self.model(cat)

        # Divide embedding into support and query
        z_prototypes = tf.reshape(z[:n_class * n_support], [n_class, n_support, z.shape[-1]])
        # Prototypes are means of n_support examples
        z_prototypes = tf.math.reduce_mean(z_prototypes, axis=1)
        z_query = z[n_class * n_support:]
        qry_logits = z_logits[n_class * n_support:]

        # Calculate distances between query and prototypes
        dists = euclidean_distance(z_query, z_prototypes)

        # Metric learning loss
        z_qry = tf.reduce_mean(tf.reshape(z_query, [n_class, n_query, z.shape[-1]]), axis=1)
        z_proto = tf.nn.l2_normalize(z_prototypes, axis=-1)
        z_qry = tf.nn.l2_normalize(z_qry, axis=-1)
        similarities = tf.einsum("ae,pe->ap", z_qry, z_proto)
        temperature = 0.5
        similarities /= temperature

        sparse_labels = tf.range(n_class)

        loss_metric = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(sparse_labels, similarities)

        return dists, qry_logits, loss_metric

# Freeze logits function
def freezeLogits(model, freeze=True):
    for layer in model.layers:
        if freeze:
            if layer.name in ['Logits_Layer']:
                layer.trainable = False
            else:
                layer.trainable = True
        else:
            layer.trainable = True

proto = Prototypical_Shubham(model)

proto_opt = tf.keras.optimizers.legacy.Adam(lr1)

proto_train_accuracy = tf.keras.metrics.Mean()

pseudoLoss = tf.keras.losses.KLDivergence()
temp = 10

n_support = n_query = 5
n_way = len(all_classes)
# num_train_episodes = 1000
num_train_episodes = 300

if os.path.exists('log.txt'):
	os.remove('log.txt')

for ep in range(num_train_episodes):

    proto_train_accuracy.reset_states()

    freezeLogits(model, freeze=True)
    #Phase1 Training
    for it in range(20):
        with tf.GradientTape() as tape:
            e_sample = extract_sample(n_way, n_support, n_query, trainx, trainy)
            x_support, x_query = get_support_query_from_extracted_sample(e_sample)

            loss, acc = proto.set_forward_loss(x_support, x_query)
            proto_train_accuracy.update_state(acc)

            gradients = tape.gradient(loss, model.trainable_variables)
            proto_opt.apply_gradients(zip(gradients, model.trainable_variables))

            print('\rEp {:d} Iter {:d} -- Loss: {:.4f} Proto Train Acc: {:.4f}'.format(ep+1, it + 1, loss, proto_train_accuracy.result()), end="")

    freezeLogits(model, freeze=False)
    #Phase2 Training
    for it in range(20):
        with tf.GradientTape() as tape:
            e_sample = extract_sample(n_way, n_support, n_query, trainx, trainy)
            x_support, x_query = get_support_query_from_extracted_sample(e_sample)
            dists, qry_logits, loss_metric = proto.predPseudo(x_support, x_query)

            pseudo_logits = tf.nn.softmax(-1/dists/temp, axis=-1)
            qry_logits = tf.nn.softmax(qry_logits, axis=-1)
            loss = pseudoLoss(pseudo_logits, qry_logits) + loss_metric
            gradients = tape.gradient(loss, model.trainable_variables)
            proto_opt.apply_gradients(zip(gradients, model.trainable_variables))
            print('\rEp {:d} Iter {:d} -- Loss: {:.4f} Proto Train Acc: {:.4f}'.format(ep+1, it + 1, loss, proto_train_accuracy.result()), end="")
    model.save(f"{proto_samples_per_class}_Models/{ep + 1}_{proto_samples_per_class}_Shot_Ova_OnlyProto_v2.h5")



