🎓 Placement Prediction System

📌 Project Overview
This project predicts whether a student will be placed based on academic performance, skills, and other factors using Machine Learning and Deep Learning techniques.

🛠️ Approach Used

1️⃣ Data Loading & Preprocessing
Imported required libraries: pandas, sklearn, tensorflow, seaborn, matplotlib.
Loaded dataset: Placement_Prediction_data.csv.
Dropped unnecessary columns: Unnamed: 0, StudentId.
Encoded categorical features (Internship, Hackathon, PlacementStatus) using LabelEncoder.
Scaled numerical features using MinMaxScaler for normalization.

2️⃣ Exploratory Data Analysis (EDA)
Displayed dataset samples using .head().
Plotted correlation heatmap to analyze feature relationships.
Visualized placement status distribution using sns.countplot().

3️⃣ Data Splitting
Separated features (X) and target (y).
Split data into 80% training & 20% testing using train_test_split().

4️⃣ Model Training & Evaluation

✅ Logistic Regression
Trained a logistic regression model using LogisticRegression().
Evaluated using accuracy and classification report.

✅ Random Forest Classifier
Trained using RandomForestClassifier(n_estimators=100).
Evaluated using accuracy and classification report.

✅ Neural Network (Deep Learning Approach)
Built a Sequential model with:
64 neurons (ReLU activation).
32 neurons (ReLU activation).
1 output neuron (Sigmoid activation for binary classification).
Compiled with adam optimizer & binary_crossentropy loss.
Trained for 37 epochs with a batch size of 32.
Evaluated on test data.

5️⃣ Final Evaluation & Conclusion
Compared model performances to determine the best approach.
Neural Network achieved high accuracy, but Random Forest & Logistic Regression also performed well.
This model can assist universities & recruiters in predicting student placements efficiently. 🚀

📌 Key Libraries Used
Pandas, NumPy (Data Handling).
Scikit-learn (ML Models & Preprocessing).
TensorFlow/Keras (Neural Networks).
Matplotlib, Seaborn (Data Visualization).
