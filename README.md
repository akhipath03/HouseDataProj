# New Jersey Residential Price Projection System

## Project Overview
This project implements a residential price prediction system for properties in New Jersey. It uses machine learning models to predict house prices based on various features such as the number of bedrooms, bathrooms, lot size, house size, and zip code.

## Technologies Used
- Python 3.9
- pandas: For data manipulation and analysis
- scikit-learn: For machine learning models and preprocessing
- TensorFlow/Keras: For building and training neural networks
- NumPy: For numerical computing

## Project Structure
The project consists of three main Python scripts:

1. `DataCleaning.py`: Handles data preprocessing and cleaning
2. `Neural_Model.py`: Implements a neural network model for price prediction
3. `RFR_Model.py`: Implements a Random Forest Regressor model for price prediction

## Data Cleaning (DataCleaning.py)
This script performs the following operations:
- Loads the original dataset
- Filters data for New Jersey
- Removes unnecessary columns
- Handles missing values
- Removes outliers
- Processes zip codes
- Saves the cleaned data to a new CSV file

## Neural Network Model (Neural_Model.py)
This script:
- Loads the processed data
- Preprocesses features (one-hot encoding for zip codes, normalization for numerical features)
- Splits data into training and validation sets
- Defines and trains a neural network model
- Evaluates the model's performance
- Provides a sample prediction

## Random Forest Regressor Model (RFR_Model.py)
This script:
- Loads the processed data
- Preprocesses features similarly to the neural network model
- Performs hyperparameter tuning using RandomizedSearchCV
- Trains a Random Forest Regressor with the best parameters
- Evaluates the model's performance
- Provides a sample prediction

## How to Use
1. Ensure all required libraries are installed:
   ```
   pip install pandas scikit-learn tensorflow numpy
   ```

2. Run the data cleaning script:
   ```
   python DataCleaning.py
   ```

3. Run either the Neural Network model or the Random Forest model:
   ```
   python Neural_Model.py
   ```
   or
   ```
   python RFR_Model.py
   ```

## Results
Both models provide metrics such as Mean Absolute Error, Mean Squared Error, and R-squared score to evaluate their performance. The actual values will depend on the specific dataset and training process.

## Future Improvements
- Experiment with feature engineering to potentially improve model performance
- Implement cross-validation for more robust evaluation
- Create a user interface for easy price predictions
- Regularly update the dataset to maintain model relevance
