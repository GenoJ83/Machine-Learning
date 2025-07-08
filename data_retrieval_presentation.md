# Data Retrieval Methods and Residual Analysis

## Table of Contents
1. Introduction
2. SQL Databases
3. NoSQL Databases
4. APIs
5. Cloud Data Sources
6. Practical Examples
7. Residual Analysis
8. Best Practices

## 1. Introduction
- Data retrieval is the process of accessing and extracting data from various sources
- Different data sources require different approaches and technologies
- Important considerations:
  - Data volume
  - Real-time requirements
  - Data consistency
  - Security

## 2. SQL Databases
### Common SQL Databases
- MySQL
- PostgreSQL
- Oracle
- Microsoft SQL Server

### Example: Python MySQL Connection
```python
import mysql.connector

# Establish connection
def connect_mysql():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='mydatabase',
            user='root',
            password='password'
        )
        return connection
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        return None

# Query example
def get_data():
    connection = connect_mysql()
    if connection:
        cursor = connection.cursor()
        query = """
        SELECT * FROM users
        WHERE age > 18
        ORDER BY created_at DESC
        LIMIT 10
        """
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        return results
```

## 3. NoSQL Databases
### Types of NoSQL Databases
1. Document-based (MongoDB)
2. Key-value (Redis)
3. Column-family (Cassandra)
4. Graph (Neo4j)

### MongoDB Example
```python
from pymongo import MongoClient

def connect_mongodb():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['mydatabase']
    return db

def find_users():
    db = connect_mongodb()
    users = db.users.find({
        "age": {"$gt": 18},
        "status": "active"
    }).sort("created_at", -1).limit(10)
    return list(users)
```

## 4. APIs
### Common API Types
- RESTful APIs
- GraphQL
- SOAP
- gRPC

### REST API Example
```python
import requests

def get_user_data(user_id):
    url = f"https://api.example.com/users/{user_id}"
    headers = {
        'Authorization': 'Bearer your_token',
        'Content-Type': 'application/json'
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code}")

# Pagination example
def get_all_users():
    all_users = []
    page = 1
    
    while True:
        url = f"https://api.example.com/users?page={page}&limit=50"
        response = requests.get(url)
        
        if response.status_code != 200:
            break
            
        users = response.json()
        if not users:
            break
            
        all_users.extend(users)
        page += 1
        
    return all_users
```

## 5. Cloud Data Sources
### Popular Cloud Services
- AWS S3
- Google Cloud Storage
- Azure Blob Storage
- Snowflake
- BigQuery

### AWS S3 Example
```python
import boto3

def get_s3_data(bucket_name, key):
    s3 = boto3.client('s3')
    try:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        data = response['Body'].read()
        return data
    except Exception as e:
        print(f"Error accessing S3: {e}")
        return None

# Using AWS SDK with pagination
def list_s3_objects(bucket_name):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page['Contents']:
            print(obj['Key'])
```

## 6. Practical Examples

### Example 1: Multi-source Data Retrieval
```python
# Combining data from SQL and NoSQL
import mysql.connector
from pymongo import MongoClient

# Get data from MySQL
def get_sql_data():
    connection = mysql.connector.connect(host='localhost', database='orders')
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM orders")
    sql_data = cursor.fetchall()
    cursor.close()
    connection.close()
    return sql_data

# Get data from MongoDB
def get_mongo_data():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['inventory']
    return list(db.products.find())

# Combine data
def get_combined_data():
    sql_data = get_sql_data()
    mongo_data = get_mongo_data()
    
    # Process and combine data as needed
    return {
        "orders": sql_data,
        "products": mongo_data
    }
```

### Example 2: API Integration with Caching
```python
import requests
from datetime import datetime, timedelta
from functools import lru_cache

@lru_cache(maxsize=128)
def get_cached_user_data(user_id):
    url = f"https://api.example.com/users/{user_id}"
    response = requests.get(url)
    return response.json()

# Rate limiting example
def get_user_data_with_rate_limit(user_id):
    last_request = datetime.now()
    min_interval = timedelta(seconds=1)
    
    current_time = datetime.now()
    if current_time - last_request < min_interval:
        time.sleep((min_interval - (current_time - last_request)).total_seconds())
    
    return get_cached_user_data(user_id)
```

## 7. Best Practices

### Security Considerations
- Always use secure connections (HTTPS)
- Implement proper authentication
- Use environment variables for sensitive data
- Implement rate limiting
- Use proper error handling

### Performance Optimization
- Use indexes appropriately
- Implement caching strategies
- Use pagination for large datasets
- Optimize queries
- Consider data compression

### Error Handling
```python
# Generic error handling example
def safe_data_retrieval(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConnectionError:
            print("Connection error occurred")
            return None
        except TimeoutError:
            print("Request timed out")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None
    return wrapper
```

### Monitoring and Logging
- Implement proper logging
- Monitor performance metrics
- Track error rates
- Set up alerts for critical issues
- Regularly audit data access patterns

## 7. Residual Analysis
### What are Residuals?
- Definition: The difference between observed and predicted values
- Formula: Residual = Observed - Predicted
- Key types:
  - Raw residuals
  - Standardized residuals
  - Studentized residuals
  - Pearson residuals

### Residual Analysis in Practice
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example data
def calculate_residuals(X, y):
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get predictions
    y_pred = model.predict(X)
    
    # Calculate residuals
    residuals = y - y_pred
    
    return residuals, y_pred

# Plotting residuals
def plot_residuals(X, y):
    residuals, y_pred = calculate_residuals(X, y)
    
    # Residual plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    
    # Normal probability plot
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Example usage
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])
plot_residuals(X, y)
```

### Residual Analysis for Model Evaluation
1. **Checking Model Assumptions**
   - Linearity
   - Independence
   - Homoscedasticity
   - Normality

2. **Common Issues Indicated by Residuals**
   - Non-linear relationships
   - Outliers
   - Heteroscedasticity
   - Autocorrelation

3. **Residual Analysis in Machine Learning**
```python
from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_model_residuals(y_true, y_pred):
    # Calculate basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Calculate residual statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Create residual analysis report
    analysis = {
        'MSE': mse,
        'RMSE': rmse,
        'Mean Residual': mean_residual,
        'Std Dev Residual': std_residual,
        'Residual Distribution': {
            'Skewness': residuals.skew(),
            'Kurtosis': residuals.kurtosis()
        }
    }
    
    return analysis

# Example usage
def analyze_model_performance(X, y, model):
    y_pred = model.predict(X)
    analysis = evaluate_model_residuals(y, y_pred)
    
    print("Model Performance Analysis:")
    for metric, value in analysis.items():
        if isinstance(value, dict):
            print(f"\n{metric}:")
            for sub_metric, sub_value in value.items():
                print(f"  {sub_metric}: {sub_value:.4f}")
        else:
            print(f"{metric}: {value:.4f}")
```

### Advanced Residual Analysis
1. **Cook's Distance**
```python
from statsmodels.stats.outliers_influence import OLSInfluence

def calculate_cooks_distance(X, y):
    model = LinearRegression()
    model.fit(X, y)
    
    # Get influence measures
    influence = OLSInfluence(model)
    
    # Calculate Cook's distance
    cooks_distance = influence.cooks_distance[0]
    
    return cooks_distance
```

2. **Leverage Statistics**
```python
def calculate_leverage(X):
    # Add intercept if not present
    if not np.all(X[:, 0] == 1):
        X = np.column_stack((np.ones(X.shape[0]), X))
    
    # Calculate hat matrix
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    
    # Leverage values are the diagonal of H
    leverage = np.diag(H)
    
    return leverage
```

### Best Practices for Residual Analysis
1. Always plot residuals
2. Check for patterns in residual plots
3. Use standardized residuals for comparison
4. Consider residual autocorrelation
5. Implement robust error metrics
6. Regularly update residual analysis procedures
