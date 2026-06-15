# **Transformer-Based fMRI Analysis for Autism Prediction**   

I’ve been working on an app that applies a **Transformer-based fMRI encoder model** to predict autism from resting-state fMRI data. The model is trained on datasets like **ABIDE** and integrates **brain connectivity features with demographic data (age, gender)** for classification.  

## **How it works:**  
📡 **Input:** Preprocessed fMRI data (CPAC or similar), age, and gender.  
🔗 **Feature Extraction:** Computes functional connectivity matrices using Nilearn.  
🧠 **Model:** A transformer-based architecture processes the extracted features, leveraging **multi-head attention** and **learned embeddings** for age and gender.  
📊 **Output:** A probability score indicating the likelihood of autism, along with a visualization.  

The model was trained with:  
✔ **BCEWithLogitsLoss** (handling class imbalance)  
✔ **Ranger optimizer** (RAdam + Lookahead)  
✔ **Cosine Annealing LR scheduling**  
✔ **Pooling strategies** (mean, max, attention)  

🔗 **Try it out here:** [https://huggingface.co/spaces/JayLacoma/fMRI-ASD-Classifier](https://huggingface.co/spaces/JayLacoma/fMRI-ASD-Classifier)  

