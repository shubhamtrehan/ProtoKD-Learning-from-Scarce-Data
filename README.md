[ProtoKD: Learning from Extremely Scarce Data for Parasite Ova Recognition](https://arxiv.org/abs/2309.10210v1)

The code focuses on applying prototypical networks with a Wide Residual Network backbone for few-shot learning tasks, incorporating a two-phase training scheme with pseudo-labeling and metric learning.
![image](https://github.com/user-attachments/assets/eebd36c4-d38d-44c0-9021-266fcd53eb40)

**Overview**

The repository includes the following components:

Prototypical Network Training:
Implements a few-shot learning framework where a prototypical network is trained using episodic training. The training script (protokd.py) includes two phases:

Phase 1: Standard episodic training where support and query sets are sampled and the prototypical loss is computed.

Phase 2: Pseudo-labeling combined with metric learning loss to refine the feature representations.

Model Backbone:
The feature extractor is based on a Wide Residual Network architecture defined in wide_resnet.py and assembled in models.py. This network is used to generate feature embeddings that are compared using Euclidean distance to form prototypes.

Data Handling:
The dataLoader_trial.py file defines the OvaDataLoader class for loading, preprocessing, and augmenting the OVA dataset. This class facilitates the episodic data sampling required for few-shot learning.

Evaluation and Visualization:
The protokd_precision_recall_heatmap.py script generates visualizations including a table of class-wise precision, recall, F1-scores, and a confusion matrix heatmap to evaluate model performance.

Current implementation is a baseline where standard rotation augmention is utilized. For the exact implementation of our paper, extract_sample() function needs to be swapped with the OvaDataLoader class
