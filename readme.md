## Overview

This repository contains code and resources for project: multi-scale geomechanics: Machine learning from a mesoscale perspective.
In this internship, we aim to use the power of Artificial Neural Networks (ANNs) to simulate the behavior of mesostructures within granular materials. 
By replacing the conventional 2D H-model framework with machine learning models, we aim to not only expedite computational efficiency but also gain valuable insights into the complexities of granular material behaviors.
In this README, you'll find information about the project, how to set up the environment, and how to run the code. 

## Project Overview

In the pursuit of our mission, we have outlined the main tasks and perspectives for this internship project:

**Main Tasks:**

1. **Generate a Training Database:** We will create a comprehensive training database using the analytical description of the standard H-cell. This database will serve as the foundation for training our machine learning models.

2. **Train Deep Neural Networks:** Our primary goal is to train deep neural networks to replace the analytical relationships traditionally used for the H-cell. We will then compare the computational efficiency of these models at the macroscale across different loading paths.


## Installation

To set up this project locally, follow these steps:

1. Clone the repository to your local machine:

2. Install project dependenciesrequirements.txt file

	pip install -r requirements.txt

## Code Structure

The project's codebase is organized into several directories and follows a modular structure to enhance readability and maintainability. 

Here's an overview of the key components and their functions:

1. data/

The data/ directory is reserved for dataset storage. 
It includes subdirectories for the training, validation, and test datasets, as well as any additional data required for the project. 
Because the training files are too large to upload conveniently, if you don't have the data files, please run Data_generation.py, Data_process.py, and Data_normalization.py in sequence to generate the data.

2. models/

In the models_and_training.py file, we have retained all machine learning models and the training process. 
When using them, please import the data files or adjust the training dataset path accordingly.


3. tests/

In Model_test.py, we conducted performance testing of all models on the entire dataset. 
This serves as a basis for subsequent selection and potential optimization.

4. output/

In the \output directory, we have selected a well-performing model with data normalization and a well-performing model without data normalization for comparison during testing.
The selected models will replace the original interpretable models in some tests for analysis and comparison.

Note that All the .py files are currently derived from the .ipynb files, and we will gradually add annotations to improve readability.