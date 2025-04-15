
# 🏠 Buenos Aires Apartment Price Prediction

This project uses data science and machine learning techniques to predict **apartment prices** in **Capital Federal, Buenos Aires 🇦🇷**. The goal is to build a reliable regression model that estimates property prices based on key features such as neighborhood, latitude and longitude, and surface area (m²).

---

## 📌 Project Overview

Real estate pricing is influenced by a variety of factors including location, number of rooms, surface area, and property type. This project includes the following steps:

- Data cleaning & preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model training & evaluation  
- Price prediction using machine learning  

---

## 🧰 Tools & Technologies

- **Python**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn**
- **Jupyter Notebook**
- **Git & GitHub**

---

## 📊 Dataset

The dataset consists of real estate listings from **Buenos Aires** and includes the following key features:

- Property Type (house, apartment, PH, store)
- Neighborhood (57 neighborhoods included)
- Surface Area (m²)
- Number of rooms
- Price in USD

**Filtered to only include:**

- Apartments only (`property_type == "apartment"`)  
- Properties located in **Capital Federal**  
- Price range capped at **$500,000**  
- Removed outliers based on surface area  
- Dropped high-cardinality and highly correlated features  

> **Source:** Kaggle

---

## 🔍 Project Steps

### 1. Data Cleaning
- Handled missing values using imputation
- Converted incorrect data types
- Removed outliers based on surface area
- Eliminated features with high or low cardinality
- Addressed multicollinearity issues

### 2. Exploratory Data Analysis (EDA)
- Analyzed price distribution by neighborhood  
- Explored apartment size trends  
- Identified key factors influencing price  

### 3. Feature Engineering
- Categorical feature (`neighbourhood`) encoded using **OneHotEncoder**  
- Numerical features cleaned and standardized  
- Pipeline created to handle preprocessing steps automatically  

### 4. Model Building
- Trained and evaluated regression models, including:
  - **Linear Regression** (baseline)
  - **Ridge Regression** (final model)
- Model performance evaluated using **Mean Absolute Error (MAE)**

### 5. Prediction
- Built an end-to-end **pipeline** that accepts apartment features and predicts price  
- Pipeline includes:
  - OneHotEncoding for categorical variables
  - Imputation of missing values
  - Ridge Regression for prediction


## 🚀 Deployment Instructions (Local)

Follow these steps to run the project locally on your machine:

### 1. Clone the Repository

Download the project from GitHub:

```bash
git clone https://github.com/your-username/buenos-aires-price-prediction.git
cd buenos-aires-price-prediction
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### 3. Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not available, manually install the core packages:
> `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`

### 4. Run the Jupyter Notebook

Launch the notebook environment:

```bash
jupyter notebook
```

Then open the main notebook (e.g., `buenos_aires_price_prediction.ipynb`) to explore the data, run the analysis, and make predictions.




