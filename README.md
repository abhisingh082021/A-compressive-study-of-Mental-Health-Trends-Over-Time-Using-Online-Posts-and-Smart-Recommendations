# A-compressive-study-of-Mental-Health-Trends-Over-Time-Using-Online-Posts-and-Smart-Recommendations


AI-Based Mental Health Clustering
This code automatically groups mental-health-related tweets into categories like stress, anxiety, depression, or neutral by converting text into vector embeddings and applying clustering. It first loads and cleans the dataset, merges all important text fields into one column, and generates sentence embeddings using a fast transformer model. Then it uses K-Means clustering to separate tweets into four groups without needing labels. Finally, it auto-assigns cluster names using keyword matching and saves the results.

Important Points

Data Loading & Cleaning: Reads the CSV file, removes duplicates, fills missing values.

Text Merging: Combines tweet, username, name, and place into a single “text” column.

Embedding Generation: Uses all-MiniLM-L6-v2 model to convert text into numerical vectors.

Unsupervised Clustering: Applies K-Means to form four mental-health clusters.

Cluster Naming: Automatically assigns cluster labels (stress, anxiety, depression, neutral) using keyword matching.

Saving Output: Exports the final predictions back into the CSV file.

------------------------------

AI-Driven Mental Health Prediction Model
This code builds a complete machine-learning pipeline to classify mental-health related tweets using multiple models and compares which one performs best. It loads and cleans the dataset, merges text from different columns, and creates labels either from the dataset or through keyword heuristics. The text is then converted into numerical features using both TF-IDF and transformer-based embeddings. Several models like Logistic Regression, SVM, and Random Forest are trained and evaluated on accuracy, precision, recall, and F1-score. A bar graph is generated to show which model performs the best, and finally, the chosen model predicts labels for the entire dataset and saves the results.

Smart Data Cleaning: Removes missing values, duplicates, and merges tweet-related text into a single column.

Auto-Labeling: If no label exists, it assigns labels using mental-health keywords.

Multiple Text Representations: Uses TF-IDF features and transformer embeddings (all-MiniLM-L6-v2).

Model Training: Trains four models — Logistic Regression, LinearSVC, Random Forest — on different vector types.

Fair Evaluation: Compares all models using accuracy, precision, recall, and F1-score.

Graph Visualization: Plots a bar chart to show which model performs best.

Best Model Selection: Automatically picks the top-performing model based on your chosen metric.

Final Output: Adds predictions to the dataset and saves the updated CSV file.


Loaded: (3082, 41)
Using text column: tweet
No label column found — creating heuristic labels (keyword-based).
After cleaning: (3082, 42)
Train/Test sizes: 2465 617
Fitting TF-IDF...
Training TF-IDF models...
Computing embeddings (this may take a moment)...
Batches: 100%
 39/39 [01:36<00:00,  1.07it/s]
Batches: 100%
 10/10 [00:16<00:00,  1.19s/it]

TFIDF + LogisticRegression -> acc:0.9984 prec:1.0000 rec:0.9983 f1:0.9992

TFIDF + LinearSVC -> acc:0.9984 prec:1.0000 rec:0.9983 f1:0.9992

Embeddings + LogisticRegression -> acc:0.9984 prec:1.0000 rec:0.9983 f1:0.9992

Embeddings + RandomForest -> acc:0.9935 prec:0.9933 rec:1.0000 f1:0.9967

Summary:
                                  accuracy  precision    recall        f1
model                                                                   
TFIDF + LogisticRegression       0.998379   1.000000  0.998322  0.999160
TFIDF + LinearSVC                0.998379   1.000000  0.998322  0.999160
Embeddings + LogisticRegression  0.998379   1.000000  0.998322  0.999160
Embeddings + RandomForest        0.993517   0.993333  1.000000  0.996656


<img width="974" height="584" alt="image" src="https://github.com/user-attachments/assets/f89b1dd9-2788-4b90-a0b0-145babd310b9" />


-------------------------


# **ML Model Comparison With Line Graph**

This code performs a complete comparison of multiple machine-learning models to identify which algorithm predicts mental-health related text most accurately. It loads and cleans the dataset, automatically creates labels using mental-health keywords if none exist, and splits the text into training and testing sets. Two types of features—TF-IDF vectors and transformer embeddings—are generated, and four different classifiers are trained. Each model’s accuracy, precision, recall, and F1-score are calculated, and a line graph is drawn to visually compare their performance. Finally, the model with the highest F1-score is selected automatically as the best algorithm.

- Data Preprocessing: Loads CSV, removes duplicates, prepares a clean text column.

- Auto Labeling: Creates labels using mental-health keywords when missing.

- Feature Engineering: Converts text into TF-IDF and MiniLM transformer embeddings.

- Model Training: Logistic Regression, Linear SVC, and Random Forest models are trained with both vector types.

- Evaluation Metrics: Computes accuracy, precision, recall, and F1-score for every model.

- Line Graph: Plots all model metrics for easy visual comparison.

- Best Model Selection: Automatically picks the algorithm with the highest F1-score.

Accuracy  Precision    Recall        F1
Model                                                                   
TFIDF + LogisticRegression       0.952846   0.971619  0.979798  0.975692
TFIDF + LinearSVC                0.952846   0.970050  0.981481  0.975732
Embeddings + LogisticRegression  0.886179   0.981618  0.898990  0.938489
Embeddings + RandomForest        0.965854   0.965854  1.000000  0.982630


<img width="980" height="584" alt="image" src="https://github.com/user-attachments/assets/0dc005b8-492b-47bb-a1b1-6c9c8ba21578" />


-------------------------------

Future Work: Advanced Mental-Health Trend & Recommendation Pipeline
This advanced script builds a complete, high-speed AI pipeline that loads and cleans mental-health text data, generates sentence-level embeddings using a transformer model, reduces their dimensions for speed, trains multiple machine-learning models, compares their performance, analyzes monthly mental-health trends, and finally creates a smart recommendation system to suggest similar posts. It uses caching, PCA, LightGBM (if available), TF-IDF, logistic regression, and nearest-neighbor search to ensure the pipeline is both fast and scalable.

Optimized Data Loading: Cleans text, handles missing values, and auto-generates labels if not provided.

Transformer Embeddings: Converts text into numerical vectors using SBERT with caching for fast re-runs.

Dimensionality Reduction: Applies PCA to speed up model training and reduce memory usage.

Multi-Model Training: Trains TF-IDF + Logistic, Embeddings + Logistic, and Embeddings + LightGBM (or Random Forest).

Performance Ranking: Evaluates all models using accuracy, precision, recall, and F1 to select the best one.

Trend Analysis: Generates smooth 3-month rolling trend graphs based on predicted labels over time.

Recommendation Engine: Uses Nearest Neighbors on reduced embeddings to find similar mental-health posts.

Artifacts Saved: Stores models, embeddings, plots, and metrics for future deployment and analysis.


 Loading dataset... done. Rows: 3082
After cleaning: (3082, 44)
Cleaning text (vectorized)... done.
modules.json: 100% 349/349 [00:00<00:00, 16.0kB/s]config_sentence_transformers.json: 100% 116/116 [00:00<00:00, 6.59kB/s]README.md:  10.5k/? [00:00<00:00, 479kB/s]sentence_bert_config.json: 100% 53.0/53.0 [00:00<00:00, 2.75kB/s]config.json: 100% 612/612 [00:00<00:00, 30.2kB/s]model.safetensors: 100% 90.9M/90.9M [00:01<00:00, 90.8MB/s]tokenizer_config.json: 100% 350/350 [00:00<00:00, 33.9kB/s]vocab.txt:  232k/? [00:00<00:00, 6.03MB/s]tokenizer.json:  466k/? [00:00<00:00, 18.9MB/s]special_tokens_map.json: 100% 112/112 [00:00<00:00, 10.6kB/s]config.json: 100% 190/190 [00:00<00:00, 12.0kB/s]Encoding texts to embeddings (batched)...
Batches: 100% 49/49 [00:45<00:00,  2.05it/s]Embeddings saved to cached_embeddings.npy — time 45.6s
Embeddings reduced -> shape: (3082, 128)
Train/Test sizes: 2465 617
Training TF-IDF + LogisticRegression (fast)... /usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
done.
Training Embeddings + LogisticRegression... /usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
done.
Training Embeddings + LightGBM (fast)... done.

Model comparison:
                 accuracy  precision    recall        f1
model                                                  
TFIDF+Logistic  0.998379   1.000000  0.998322  0.999160
Emb+Logistic    0.998379   1.000000  0.998322  0.999160
Emb+LightGBM    0.998379   0.998325  1.000000  0.999162
Saved model comparison plot -> results_fast/model_comparison_line.png
Best model by F1: Emb+LightGBM
Saved trend graph -> results_fast/trend_monthly_smoothed.png
Sample recommendations (fast):
- idx:2812 | dist:0.426 | label:1 | text: i feel lonely and weird sosadtoday so sad today
- idx:12 | dist:0.458 | label:1 | text: live with depression and anxiety no motivation to leave your bed dread leave your house not be able to go out unable to 
- idx:2830 | dist:0.483 | label:1 | text: i want sex and to be alone sosadtoday so sad today
- idx:2960 | dist:0.488 | label:1 | text: meet me at the corner of insomnia and difficulty live in the world sosadtoday so sad today
- idx:2779 | dist:0.523 | label:1 | text: mentally i be always in bed sosadtoday so sad today
Artifacts saved to results_fast
DONE — Fast pipeline executed.
