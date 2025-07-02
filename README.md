# Fake News Detection Using LSTM

## Overview
This project implements a Long Short-Term Memory (LSTM) neural network to detect fake news by classifying articles as "Real" or "Fake." The notebook `Fake_News_LSTM.ipynb` processes two datasets (`True.txt` and `Fake.txt`), performs text preprocessing, trains an LSTM model using Keras, and evaluates its performance.

## Objective
The goal is to build a binary classification model to distinguish between real and fake news articles based on their text content, leveraging LSTM for sequence modeling.

## Dataset
The dataset consists of two CSV files located in the `Fake_news_detection` directory:
- `True.txt`: Contains real news articles with columns `title`, `text`, `subject`, and `date`.
- `Fake.txt`: Contains fake news articles with the same columns.
- A `target` column is added to label real articles as `1` and fake articles as `0`.

## Dependencies
The project requires the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `nltk`
- `re`
- `sklearn`
- `keras`

Install dependencies using:
```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn keras
```

Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Notebook Structure
1. **Data Loading and Preprocessing**:
   - Loads `True.txt` and `Fake.txt` using `pandas`.
   - Adds a `target` column: `1` for real news, `0` for fake news.
   - Merges the datasets into a single DataFrame (`raw_data`).
   - Checks for null values (none found in the dataset).
   - Combines `title` and `text` columns for analysis (not explicitly shown but implied for preprocessing).
   - Applies text preprocessing using `nltk` and `re` for tokenization, stopword removal, and cleaning (e.g., removing punctuation).

2. **Model Preparation**:
   - Uses `keras.preprocessing.text` and `sequence` for text tokenization and sequence padding.
   - Splits the data into training and testing sets using `sklearn.model_selection.train_test_split`.

3. **Model Architecture**:
   - Builds a `Sequential` Keras model with:
     - An `Embedding` layer to convert words into dense vectors.
     - An `LSTM` layer to capture sequential dependencies in text.
     - A `Dropout` layer to prevent overfitting.
     - A `Dense` layer with a sigmoid activation for binary classification.
   - Note: The exact model architecture and hyperparameters are not shown in the provided notebook snippet.

4. **Training and Evaluation**:
   - Trains the model and plots training and validation loss using `matplotlib`.
   - Evaluates the model on the test set using `classification_report` from `sklearn.metrics`.
   - The model predicts labels (`0` for Fake, `1` for Real) and reports precision, recall, F1-score, and accuracy.
   - Sample output shows:
     - Accuracy: 54%
     - High recall for Fake (1.00) but poor performance for Real (0.00 recall), indicating potential issues with model balance or training.

## Output
- **Loss Plot**: Visualizes training and validation loss over epochs using `matplotlib`.
- **Classification Report**: Provides precision, recall, F1-score, and support for Fake and Real classes.
- The provided results suggest the model struggles to classify real news, possibly due to class imbalance or insufficient training.

## Usage
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Place `True.txt` and `Fake.txt` in the `Fake_news_detection` directory.
3. Update file paths in the notebook to match your local environment (e.g., `C:\Users\ohkba\OneDrive\Documents\Fake_news_detection`).
4. Run the notebook cells sequentially to:
   - Load and preprocess the data.
   - Train the LSTM model.
   - Evaluate and visualize the results.

## Notes
- **Model Performance**: The classification report indicates poor performance for the "Real" class (0% recall), suggesting the model may be biased toward predicting "Fake." This could be due to class imbalance, insufficient preprocessing, or suboptimal hyperparameters.
- **File Paths**: The notebook uses hardcoded paths (e.g., `C:\Users\ohkba\OneDrive\Documents\Fake_news_detection`). Update these to match your directory structure.
- **Python Version**: The notebook assumes Python 3.9. Ensure compatibility with your environment.
- **Missing Code**: The notebook snippet does not include the full preprocessing and model-building code. Ensure the complete code includes tokenization, sequence padding, and model compilation steps.
- **Keras Dependency**: The notebook uses Keras for the LSTM model. Ensure compatibility with your TensorFlow/Keras version.

## Future Improvements
- Address class imbalance using techniques like oversampling (SMOTE) or class weights.
- Enhance text preprocessing (e.g., lemmatization, stemming, or TF-IDF features).
- Experiment with hyperparameters (e.g., LSTM units, dropout rate, embedding dimensions).
- Add cross-validation to improve model robustness.
- Incorporate additional features (e.g., `subject` or `date`) for better context.
- Visualize confusion matrix or ROC curves for deeper performance analysis.

## License
This project is licensed under the MIT License.
