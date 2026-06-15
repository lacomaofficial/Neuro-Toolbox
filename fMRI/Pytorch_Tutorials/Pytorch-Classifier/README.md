# Functional Connectivity Classifier 

This project aims to classify subjects based on functional connectivity derived from fMRI data using a deep learning approach. Specifically, the project explores the classification of subjects into different gender (male vs. female) by analyzing fMRI data during movie watching. The project leverages functional connectivity matrices extracted from fMRI data as input features for a neural network classifier. The experiment uses the Developmental fMRI Dataset, which consists of fMRI scans of children (ages 3-13) and young adults (ages 18-39) while they watched movies. This dataset is ideal for demonstrating how machine learning models can be trained to classify participants based on brain connectivity patterns.

## Results

### 1. **Superior Temporal Sulcus with Angular Gyrus – Superior Parietal Lobule Anterior** 

This connection between the **STS/Angular Gyrus** and the **Superior Parietal Lobule Anterior** could reflect the brain's integration of **social information with spatial awareness**. During the film, this connection might allow individuals to integrate **social cues** (from observing characters) with an understanding of **spatial relationships** in the scene, indicating possible differences in how males and females interpret these aspects during the film.



### 2. **Superior Parietal Lobule Posterior – Intraparietal Sulcus (Left Hemisphere)**

The **Superior Parietal Lobule Posterior** and **Left IPS** connection suggests gender differences in **visuospatial processing** and **attention to objects in space**. These regions work together to allow individuals to allocate **visual attention** and **coordinate actions** in space, possibly suggesting that males and females use different strategies for interpreting spatial scenes or navigating the visual environment during the film.



### 3. **Cerebellum I-V – Precentral Gyrus Medial**

This connection likely reflects gender differences in **motor planning** and **bodily coordination**. The **Cerebellum** and **Precentral Gyrus** are both involved in coordinating **movement and motor control**, so this connection might indicate that males and females process **motor responses** differently when reacting to film scenes, such as preparing for action or responding to physical stimuli.



### 4. **Insula Antero-Superior – Precentral Gyrus Medial** 

This connection between the **insula** and the **precentral gyrus** may point to **gender-specific differences** in how emotions influence **motor responses**. For example, females might have stronger integration of emotional states (insula) with physical expressions (precentral gyrus), leading to more pronounced emotional reactions in their motor responses, while males might show different patterns of emotional regulation and expression.


### 5. **Intraparietal Sulcus (Right Hemisphere) – Lingual Gyrus**
   
This connection suggests that males and females might differ in how they **process visual stimuli** in terms of **attention** and **object recognition**. The **right IPS** and **Lingual Gyrus** working together help with **tracking visual information**, interpreting shapes, and coordinating visual attention, so the differences in this connection could reflect how each gender processes complex visual information differently during the film, potentially affecting how visual scenes are perceived.

![image](https://github.com/user-attachments/assets/8afde7d4-9ef1-4476-94f3-6d899ea8d302)




## Experimental Setup

1. **Data Loading and Preprocessing**:
   - The fMRI data is downloaded from the Nilearn library, which fetches the Developmental fMRI dataset.
   - The data includes functional MRI Nifti images, confounds, and phenotypic information such as age, age group, gender, and handedness.

2. **Feature Extraction**:
   - Functional connectivity matrices are computed for each subject using the `ConnectivityMeasure` from Nilearn. The `MultiNiftiMapsMasker` is utilized to extract time-series data from predefined brain regions (using an atlas) and compute correlations between them to create connectivity matrices.

3. **Data Preparation for Machine Learning**:
   - The connectivity matrices are vectorized and used as input features for the classification model.
   - The target variable (e.g., gender) is encoded, and the data is split into training and testing sets.

4. **Modeling**:
   - A neural network model is defined with multiple hidden layers, batch normalization, and dropout for regularization. The architecture of the network is optimized using Optuna, an optimization framework that tunes hyperparameters like the number of hidden layers, learning rate, and dropout rate.

5. **Training and Evaluation**:
   - The model is trained using the training dataset and evaluated on the testing dataset. Metrics such as log-loss, accuracy, AUC-ROC, and Precision-Recall AUC are computed to assess the model's performance.
   - Early stopping is implemented to prevent overfitting by stopping the training process when no improvement is observed after a certain number of epochs.

6. **Feature Importance Analysis**:
   - The importance of different brain regions (features) is analyzed by perturbing each feature and measuring its impact on the model's loss. This analysis helps identify which brain regions are most critical in differentiating between classes.


## Requirements

- Python 3.x
- Libraries: `nilearn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `torch`, `optuna`, `scikit-learn`

## Summary:
- **Social Cognition and Spatial Awareness**: The connection between the **Superior Temporal Sulcus** and **Superior Parietal Lobule** highlights possible differences in how males and females process **social cues** and **spatial information**.
- **Visuospatial and Motor Processing**: Connections involving the **Superior Parietal Lobule**, **Intraparietal Sulcus**, and **Cerebellum** suggest gender-specific strategies for **spatial attention**, **movement coordination**, and **motor planning**.
- **Emotion and Movement Integration**: The **insula** and **precentral gyrus** connection indicates that gender differences in **emotional regulation** might influence **motor activity**, reflecting varying emotional responses to the film's content.
- **Visual Attention and Recognition**: The connection between the **right IPS** and **Lingual Gyrus** hints at gender-specific differences in **visual attention** and **high-level visual processing**, particularly in how visual details are interpreted and acted upon.
