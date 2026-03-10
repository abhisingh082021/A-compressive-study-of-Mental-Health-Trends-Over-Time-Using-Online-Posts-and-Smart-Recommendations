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
