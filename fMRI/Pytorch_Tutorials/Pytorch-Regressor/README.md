# Functional Connectivity Regressor

This project aims to predict age based on functional connectivity derived from fMRI data using a deep learning approach by analyzing fMRI data during movie watching. The project leverages functional connectivity matrices extracted from fMRI data as input features for a neural network regressor. The experiment uses the Developmental fMRI Dataset, which consists of fMRI scans of children (ages 3-13) and young adults (ages 18-39) while they watched movies. This dataset is ideal for demonstrating how machine learning models can be trained to classify participants based on brain connectivity patterns. 


## Results

### 1. **Precuneus Superior – Superior Fornix and Isthmus** (Importance: 0.474181)

   - **Role in Film Watching**: 
     The **precuneus** is strongly activated during reflective thinking and **self-referential processes**, such as when people relate what they see to their own experiences or emotions. During a film, this could involve **empathy**, placing oneself in the situation of a character, or daydreaming.
     The **fornix**, a memory-related structure, likely aids in recalling **past experiences** and **emotional memories**, which can influence how individuals relate to the narrative.
   
   - **Age Prediction**: 
     Age-related changes in memory networks like the **fornix** could alter how older participants recall or relate to the film's plot and characters, possibly making this connection a key feature in predicting age.

### 2. **Parieto-Occipital Sulcus Superior – Cerebellum Crus II** (Importance: 0.410588)

   - **Role in Film Watching**: 
     The **parieto-occipital sulcus** is associated with **visual-spatial processing** and understanding **scenes**. The **cerebellum**, particularly **Crus II**, has been linked to both **motor coordination** and **cognitive functions**, including **language processing** and **working memory**.
     This connection may become active during scenes requiring **visual tracking** of characters or objects moving across the screen, or interpreting spatial relationships in a scene.
   
   - **Age Prediction**: 
     As people age, the integration of visual and motor information may decline, impacting their ability to **track movement** or **process complex scenes**. This could be a strong indicator of age-related cognitive decline.

### 3. **Inferior Occipital Gyrus – Superior Parietal Lobule Anterior** (Importance: 0.394484)

   - **Role in Film Watching**: 
     The **inferior occipital gyrus** is a core region for **visual processing**, particularly in recognizing objects and faces, while the **superior parietal lobule** helps integrate this visual information with **spatial attention**.
     When watching a film, this connection likely supports the identification of characters, facial expressions, and objects in motion, as well as **directing attention** to important visual elements on the screen.
   
   - **Age Prediction**: 
     Age-related changes in visual processing and attention allocation could alter how older individuals perceive and focus on key elements in a film, making this connection relevant for predicting cognitive aging.

### 4. **Intraparietal Sulcus (Left Hemisphere) – Cuneus** (Importance: 0.370234)

   - **Role in Film Watching**: 
     The **intraparietal sulcus (IPS)** is critical for **visuospatial attention**, while the **cuneus** processes **basic visual stimuli** such as motion and orientation. Together, these regions help the brain **focus on moving objects**, track **changes in visual scenes**, and interpret **dynamic visuals** like those in a film.
   
   - **Age Prediction**: 
     As individuals age, their ability to maintain attention on dynamic stimuli (e.g., fast-paced scenes or sudden visual changes) may decline, making this connection key in predicting cognitive slowdown in older participants.

### 5. **Superior Temporal Sulcus with Angular Gyrus – Lingual Gyrus Anterior** (Importance: 0.365849)

   - **Role in Film Watching**: 
     The **superior temporal sulcus** plays a major role in **social perception**, such as recognizing **facial expressions** and **understanding intentions**. The **angular gyrus** is important for **language processing** and **semantic understanding**. The **lingual gyrus**, part of the visual cortex, processes **visual word recognition** and **scene recognition**.
     During a film, this connection would be vital for interpreting **social cues** between characters, understanding the **plot**, and linking **visual and linguistic elements**.
   
   - **Age Prediction**: 
     As social and language processing regions decline with age, older adults may show altered responses to **dialogue**, **character interactions**, or **plot comprehension**, making this connection valuable for predicting cognitive aging.

![image](https://github.com/user-attachments/assets/0f014ca2-6d00-40d0-b55a-765dfd5438fb)



## Experimental Setup

1. **Data Acquisition and Preprocessing**:
   - **Data Source**: The data used is from the `development_fmri` dataset obtained via Nilearn, containing fMRI images and associated confound variables.
   - **Data Exploration**: Initial exploration includes loading and inspecting the dataset for functional and confound files, as well as participant demographics such as age and gender.

2. **Connectivity Analysis**:
   - **Atlas Definition**: Difumo atlas is used to define 64 brain regions of interest (ROIs) for extracting functional connectivity.
   - **Connectivity Estimation**: Nilearn's `MultiNiftiMapsMasker` is employed to extract time series from ROIs and `ConnectivityMeasure` computes correlation matrices to quantify connectivity.

3. **Machine Learning Model**:
   - **Feature Selection**: Features derived from connectivity matrices are used to predict participant age.
   - **Model Architecture**: A non-linear regression model is defined using PyTorch, optimized via Optuna for hyperparameters like hidden layer sizes, dropout rates, and learning rates.
   - **Training and Evaluation**: The model is trained using `AdamW` optimizer with early stopping based on validation loss, evaluated using MSE, MAE, and R² metrics.

4. **Findings and Insights**:
   - **Neuroscientific Insights**: Significant brain regions contributing to age prediction are identified through feature importance analysis using perturbation methods.
   - **Visualization**: Visualizations such as histograms, connectivity matrices, and feature importance plots provide intuitive insights into data patterns and model outcomes.

## Requirements

- Python 3.x
- Libraries: `nilearn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `torch`, `optuna`, `scikit-learn`
- 

## Summary

1. **Functional Connectivity Patterns**:
   - Strong correlations between specific brain regions suggest coordinated activity networks.
   - Variations in connectivity strength across age groups indicate developmental changes in brain organization.

2. **Impact of Age on Connectivity**:
   - Age-related changes in connectivity metrics reflect ongoing neural maturation and functional specialization.
   - Differential patterns between children and adults highlight distinct stages of cognitive development and aging effects.

3. **Methodological Advancements**:
   - Integration of machine learning with neuroimaging allows for precise prediction of age from brain connectivity data.
   - Identification of critical brain regions underscores their roles in cognitive functions and developmental trajectories.

## Conclusion

This project demonstrates the integration of neuroimaging data analysis with machine learning techniques to uncover meaningful insights into brain connectivity and its relationship with age. By leveraging advanced tools and methodologies, it contributes to understanding neural mechanisms underlying human development and aging processes.

