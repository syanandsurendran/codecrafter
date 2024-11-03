import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulate random data for industrial operations

np.random.seed(0)
n_samples = 100

# Randomly generated inputs

labor_hours = np.random.uniform(40, 60, n_samples)
machine_downtime = np.random.uniform(0, 10, n_samples)
material_quality = np.random.uniform(0.5, 1.0, n_samples)
maintenance_frequency = np.random.uniform(0, 5, n_samples)
energy_costs = np.random.uniform(10, 20, n_samples)
production_volume = np.random.uniform(1000, 5000, n_samples)
error_rate = np.random.uniform(0, 0.2, n_samples)

# Generate outputs based on a theoretical relationship

productivity = 0.5 * labor_hours - 0.3 * machine_downtime + 0.7 * material_quality + np.random.normal(0, 1, n_samples)
cost = 1.2 * labor_hours + 0.5 * energy_costs + 0.3 * maintenance_frequency + 0.8 * machine_downtime + np.random.normal(0, 1, n_samples)
profit = 2 * productivity - 1.5 * cost + np.random.normal(0, 1, n_samples)

# Create a DataFrame

data = pd.DataFrame({
'Labor Hours': labor_hours,
'Machine Downtime': machine_downtime,
'Material Quality': material_quality,
'Maintenance Frequency': maintenance_frequency,
'Energy Costs': energy_costs,
'Production Volume': production_volume,
'Error Rate': error_rate,
'Productivity': productivity,
'Cost': cost,
'Profit': profit
})

# Define the model and features for each target variable

features = ['Labor Hours', 'Machine Downtime', 'Material Quality', 'Maintenance Frequency', 'Energy Costs']

# Model productivity

X = data[features]
y_productivity = data['Productivity']
X_train, X_test, y_train, y_test = train_test_split(X, y_productivity, test_size=0.2, random_state=42)
model_productivity = LinearRegression().fit(X_train, y_train)
productivity_pred = model_productivity.predict(X_test)
productivity_mse = mean_squared_error(y_test, productivity_pred)

# Model cost

y_cost = data['Cost']
X_train, X_test, y_train, y_test = train_test_split(X, y_cost, test_size=0.2, random_state=42)
model_cost = LinearRegression().fit(X_train, y_train)
cost_pred = model_cost.predict(X_test)
cost_mse = mean_squared_error(y_test, cost_pred)

# Model profit (uses Productivity and Cost as inputs)

y_profit = data['Profit']
X_profit = data[['Productivity', 'Cost']]
X_train, X_test, y_train, y_test = train_test_split(X_profit, y_profit, test_size=0.2, random_state=42)
model_profit = LinearRegression().fit(X_train, y_train)
profit_pred = model_profit.predict(X_test)
profit_mse = mean_squared_error(y_test, profit_pred)

print(f'Productivity MSE: {productivity_mse:.2f}')
print(f'Cost MSE: {cost_mse:.2f}')
print(f'Profit MSE: {profit_mse:.2f}')
import matplotlib.pyplot as plt

# Scatter plots to show relationships between features and outputs

plt.figure(figsize=(20, 10))

# Labor Hours vs Productivity

plt.subplot(2, 3, 1)
plt.scatter(data['Labor Hours'], data['Productivity'], color='blue', alpha=0.5)
plt.title('Labor Hours vs Productivity')
plt.xlabel('Labor Hours')
plt.ylabel('Productivity')

# Machine Downtime vs Productivity

plt.subplot(2, 3, 2)
plt.scatter(data['Machine Downtime'], data['Productivity'], color='orange', alpha=0.5)
plt.title('Machine Downtime vs Productivity')
plt.xlabel('Machine Downtime')
plt.ylabel('Productivity')

# Energy Costs vs Cost

plt.subplot(2, 3, 3)
plt.scatter(data['Energy Costs'], data['Cost'], color='green', alpha=0.5)
plt.title('Energy Costs vs Cost')
plt.xlabel('Energy Costs')
plt.ylabel('Cost')

# Productivity vs Profit

plt.subplot(2, 3, 4)
plt.scatter(data['Productivity'], data['Profit'], color='purple', alpha=0.5)
plt.title('Productivity vs Profit')
plt.xlabel('Productivity')
plt.ylabel('Profit')

# Cost vs Profit

plt.subplot(2, 3, 5)
plt.scatter(data['Cost'], data['Profit'], color='red', alpha=0.5)
plt.title('Cost vs Profit')
plt.xlabel('Cost')
plt.ylabel('Profit')

plt.tight_layout()
plt.show()

# Histograms for Productivity, Cost, and Profit

plt.figure(figsize=(15, 5))

# Productivity histogram

plt.subplot(1, 3, 1)
plt.hist(data['Productivity'], bins=10, color='skyblue', edgecolor='black')
plt.title('Productivity Distribution')
plt.xlabel('Productivity')
plt.ylabel('Frequency')

# Cost histogram

plt.subplot(1, 3, 2)
plt.hist(data['Cost'], bins=10, color='salmon', edgecolor='black')
plt.title('Cost Distribution')
plt.xlabel('Cost')
plt.ylabel('Frequency')

# Profit histogram

plt.subplot(1, 3, 3)
plt.hist(data['Profit'], bins=10, color='lightgreen', edgecolor='black')
plt.title('Profit Distribution')
plt.xlabel('Profit')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulate random data for industrial operations

np.random.seed(0)
n_samples = 100

# Randomly generated inputs

labor_hours = np.random.uniform(40, 60, n_samples)
machine_downtime = np.random.uniform(0, 10, n_samples)
material_quality = np.random.uniform(0.5, 1.0, n_samples)
maintenance_frequency = np.random.uniform(0, 5, n_samples)
energy_costs = np.random.uniform(10, 20, n_samples)
production_volume = np.random.uniform(1000, 5000, n_samples)
error_rate = np.random.uniform(0, 0.2, n_samples)

# Generate outputs based on a theoretical relationship

productivity = 0.5 * labor_hours - 0.3 * machine_downtime + 0.7 * material_quality + np.random.normal(0, 1, n_samples)
cost = 1.2 * labor_hours + 0.5 * energy_costs + 0.3 * maintenance_frequency + 0.8 * machine_downtime + np.random.normal(0, 1, n_samples)
profit = 2 * productivity - 1.5 * cost + np.random.normal(0, 1, n_samples)

# Create a DataFrame

data = pd.DataFrame({
'Labor Hours': labor_hours,
'Machine Downtime': machine_downtime,
'Material Quality': material_quality,
'Maintenance Frequency': maintenance_frequency,
'Energy Costs': energy_costs,
'Production Volume': production_volume,
'Error Rate': error_rate,
'Productivity': productivity,
'Cost': cost,
'Profit': profit
})

# Define the model and features for each target variable

features = ['Labor Hours', 'Machine Downtime', 'Material Quality', 'Maintenance Frequency', 'Energy Costs']

# Model productivity

X = data[features]
y_productivity = data['Productivity']
X_train, X_test, y_train, y_test = train_test_split(X, y_productivity, test_size=0.2, random_state=42)
model_productivity = LinearRegression().fit(X_train, y_train)
productivity_pred = model_productivity.predict(X_test)
productivity_mse = mean_squared_error(y_test, productivity_pred)

# Model cost

y_cost = data['Cost']
X_train, X_test, y_train, y_test = train_test_split(X, y_cost, test_size=0.2, random_state=42)
model_cost = LinearRegression().fit(X_train, y_train)
cost_pred = model_cost.predict(X_test)
cost_mse = mean_squared_error(y_test, cost_pred)

# Model profit (uses Productivity and Cost as inputs)

y_profit = data['Profit']
X_profit = data[['Productivity', 'Cost']]
X_train, X_test, y_train, y_test = train_test_split(X_profit, y_profit, test_size=0.2, random_state=42)
model_profit = LinearRegression().fit(X_train, y_train)
profit_pred = model_profit.predict(X_test)
profit_mse = mean_squared_error(y_test, profit_pred)

print(f'Productivity MSE: {productivity_mse:.2f}')
print(f'Cost MSE: {cost_mse:.2f}')
print(f'Profit MSE: {profit_mse:.2f}')
import matplotlib.pyplot as plt

# Scatter plots to show relationships between features and outputs

plt.figure(figsize=(20, 10))

# Labor Hours vs Productivity

plt.subplot(2, 3, 1)
plt.scatter(data['Labor Hours'], data['Productivity'], color='blue', alpha=0.5)
plt.title('Labor Hours vs Productivity')
plt.xlabel('Labor Hours')
plt.ylabel('Productivity')

# Machine Downtime vs Productivity

plt.subplot(2, 3, 2)
plt.scatter(data['Machine Downtime'], data['Productivity'], color='orange', alpha=0.5)
plt.title('Machine Downtime vs Productivity')
plt.xlabel('Machine Downtime')
plt.ylabel('Productivity')

# Energy Costs vs Cost

plt.subplot(2, 3, 3)
plt.scatter(data['Energy Costs'], data['Cost'], color='green', alpha=0.5)
plt.title('Energy Costs vs Cost')
plt.xlabel('Energy Costs')
plt.ylabel('Cost')

# Productivity vs Profit

plt.subplot(2, 3, 4)
plt.scatter(data['Productivity'], data['Profit'], color='purple', alpha=0.5)
plt.title('Productivity vs Profit')
plt.xlabel('Productivity')
plt.ylabel('Profit')

# Cost vs Profit

plt.subplot(2, 3, 5)
plt.scatter(data['Cost'], data['Profit'], color='red', alpha=0.5)
plt.title('Cost vs Profit')
plt.xlabel('Cost')
plt.ylabel('Profit')

plt.tight_layout()
plt.show()

# Histograms for Productivity, Cost, and Profit

plt.figure(figsize=(15, 5))

# Productivity histogram

plt.subplot(1, 3, 1)
plt.hist(data['Productivity'], bins=10, color='skyblue', edgecolor='black')
plt.title('Productivity Distribution')
plt.xlabel('Productivity')
plt.ylabel('Frequency')

# Cost histogram

plt.subplot(1, 3, 2)
plt.hist(data['Cost'], bins=10, color='salmon', edgecolor='black')
plt.title('Cost Distribution')
plt.xlabel('Cost')
plt.ylabel('Frequency')

# Profit histogram

plt.subplot(1, 3, 3)
plt.hist(data['Profit'], bins=10, color='lightgreen', edgecolor='black')
plt.title('Profit Distribution')
plt.xlabel('Profit')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulate random data for industrial operations

np.random.seed(0)
n_samples = 100

# Randomly generated inputs

labor_hours = np.random.uniform(40, 60, n_samples)
machine_downtime = np.random.uniform(0, 10, n_samples)
material_quality = np.random.uniform(0.5, 1.0, n_samples)
maintenance_frequency = np.random.uniform(0, 5, n_samples)
energy_costs = np.random.uniform(10, 20, n_samples)
production_volume = np.random.uniform(1000, 5000, n_samples)
error_rate = np.random.uniform(0, 0.2, n_samples)

# Generate outputs based on a theoretical relationship

productivity = 0.5 * labor_hours - 0.3 * machine_downtime + 0.7 * material_quality + np.random.normal(0, 1, n_samples)
cost = 1.2 * labor_hours + 0.5 * energy_costs + 0.3 * maintenance_frequency + 0.8 * machine_downtime + np.random.normal(0, 1, n_samples)
profit = 2 * productivity - 1.5 * cost + np.random.normal(0, 1, n_samples)

# Create a DataFrame

data = pd.DataFrame({
'Labor Hours': labor_hours,
'Machine Downtime': machine_downtime,
'Material Quality': material_quality,
'Maintenance Frequency': maintenance_frequency,
'Energy Costs': energy_costs,
'Production Volume': production_volume,
'Error Rate': error_rate,
'Productivity': productivity,
'Cost': cost,
'Profit': profit
})

# Define the model and features for each target variable

features = ['Labor Hours', 'Machine Downtime', 'Material Quality', 'Maintenance Frequency', 'Energy Costs']

# Model productivity

X = data[features]
y_productivity = data['Productivity']
X_train, X_test, y_train, y_test = train_test_split(X, y_productivity, test_size=0.2, random_state=42)
model_productivity = LinearRegression().fit(X_train, y_train)
productivity_pred = model_productivity.predict(X_test)
productivity_mse = mean_squared_error(y_test, productivity_pred)

# Model cost

y_cost = data['Cost']
X_train, X_test, y_train, y_test = train_test_split(X, y_cost, test_size=0.2, random_state=42)
model_cost = LinearRegression().fit(X_train, y_train)
cost_pred = model_cost.predict(X_test)
cost_mse = mean_squared_error(y_test, cost_pred)

# Model profit (uses Productivity and Cost as inputs)

y_profit = data['Profit']
X_profit = data[['Productivity', 'Cost']]
X_train, X_test, y_train, y_test = train_test_split(X_profit, y_profit, test_size=0.2, random_state=42)
model_profit = LinearRegression().fit(X_train, y_train)
profit_pred = model_profit.predict(X_test)
profit_mse = mean_squared_error(y_test, profit_pred)

print(f'Productivity MSE: {productivity_mse:.2f}')
print(f'Cost MSE: {cost_mse:.2f}')
print(f'Profit MSE: {profit_mse:.2f}')
import matplotlib.pyplot as plt

# Scatter plots to show relationships between features and outputs

plt.figure(figsize=(20, 10))

# Labor Hours vs Productivity

plt.subplot(2, 3, 1)
plt.scatter(data['Labor Hours'], data['Productivity'], color='blue', alpha=0.5)
plt.title('Labor Hours vs Productivity')
plt.xlabel('Labor Hours')
plt.ylabel('Productivity')

# Machine Downtime vs Productivity

plt.subplot(2, 3, 2)
plt.scatter(data['Machine Downtime'], data['Productivity'], color='orange', alpha=0.5)
plt.title('Machine Downtime vs Productivity')
plt.xlabel('Machine Downtime')
plt.ylabel('Productivity')

# Energy Costs vs Cost

plt.subplot(2, 3, 3)
plt.scatter(data['Energy Costs'], data['Cost'], color='green', alpha=0.5)
plt.title('Energy Costs vs Cost')
plt.xlabel('Energy Costs')
plt.ylabel('Cost')

# Productivity vs Profit

plt.subplot(2, 3, 4)
plt.scatter(data['Productivity'], data['Profit'], color='purple', alpha=0.5)
plt.title('Productivity vs Profit')
plt.xlabel('Productivity')
plt.ylabel('Profit')

# Cost vs Profit

plt.subplot(2, 3, 5)
plt.scatter(data['Cost'], data['Profit'], color='red', alpha=0.5)
plt.title('Cost vs Profit')
plt.xlabel('Cost')
plt.ylabel('Profit')

plt.tight_layout()
plt.show()

# Histograms for Productivity, Cost, and Profit

plt.figure(figsize=(15, 5))

# Productivity histogram

plt.subplot(1, 3, 1)
plt.hist(data['Productivity'], bins=10, color='skyblue', edgecolor='black')
plt.title('Productivity Distribution')
plt.xlabel('Productivity')
plt.ylabel('Frequency')

# Cost histogram

plt.subplot(1, 3, 2)
plt.hist(data['Cost'], bins=10, color='salmon', edgecolor='black')
plt.title('Cost Distribution')
plt.xlabel('Cost')
plt.ylabel('Frequency')

# Profit histogram

plt.subplot(1, 3, 3)
plt.hist(data['Profit'], bins=10, color='lightgreen', edgecolor='black')
plt.title('Profit Distribution')
plt.xlabel('Profit')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulate random data for industrial operations

np.random.seed(0)
n_samples = 100

# Randomly generated inputs

labor_hours = np.random.uniform(40, 60, n_samples)
machine_downtime = np.random.uniform(0, 10, n_samples)
material_quality = np.random.uniform(0.5, 1.0, n_samples)
maintenance_frequency = np.random.uniform(0, 5, n_samples)
energy_costs = np.random.uniform(10, 20, n_samples)
production_volume = np.random.uniform(1000, 5000, n_samples)
error_rate = np.random.uniform(0, 0.2, n_samples)

# Generate outputs based on a theoretical relationship

productivity = 0.5 * labor_hours - 0.3 * machine_downtime + 0.7 * material_quality + np.random.normal(0, 1, n_samples)
cost = 1.2 * labor_hours + 0.5 * energy_costs + 0.3 * maintenance_frequency + 0.8 * machine_downtime + np.random.normal(0, 1, n_samples)
profit = 2 * productivity - 1.5 * cost + np.random.normal(0, 1, n_samples)

# Create a DataFrame

data = pd.DataFrame({
'Labor Hours': labor_hours,
'Machine Downtime': machine_downtime,
'Material Quality': material_quality,
'Maintenance Frequency': maintenance_frequency,
'Energy Costs': energy_costs,
'Production Volume': production_volume,
'Error Rate': error_rate,
'Productivity': productivity,
'Cost': cost,
'Profit': profit
})

# Define the model and features for each target variable

features = ['Labor Hours', 'Machine Downtime', 'Material Quality', 'Maintenance Frequency', 'Energy Costs']

# Model productivity

X = data[features]
y_productivity = data['Productivity']
X_train, X_test, y_train, y_test = train_test_split(X, y_productivity, test_size=0.2, random_state=42)
model_productivity = LinearRegression().fit(X_train, y_train)
productivity_pred = model_productivity.predict(X_test)
productivity_mse = mean_squared_error(y_test, productivity_pred)

# Model cost

y_cost = data['Cost']
X_train, X_test, y_train, y_test = train_test_split(X, y_cost, test_size=0.2, random_state=42)
model_cost = LinearRegression().fit(X_train, y_train)
cost_pred = model_cost.predict(X_test)
cost_mse = mean_squared_error(y_test, cost_pred)

# Model profit (uses Productivity and Cost as inputs)

y_profit = data['Profit']
X_profit = data[['Productivity', 'Cost']]
X_train, X_test, y_train, y_test = train_test_split(X_profit, y_profit, test_size=0.2, random_state=42)
model_profit = LinearRegression().fit(X_train, y_train)
profit_pred = model_profit.predict(X_test)
profit_mse = mean_squared_error(y_test, profit_pred)

print(f'Productivity MSE: {productivity_mse:.2f}')
print(f'Cost MSE: {cost_mse:.2f}')
print(f'Profit MSE: {profit_mse:.2f}')
import matplotlib.pyplot as plt

# Scatter plots to show relationships between features and outputs

plt.figure(figsize=(20, 10))

# Labor Hours vs Productivity

plt.subplot(2, 3, 1)
plt.scatter(data['Labor Hours'], data['Productivity'], color='blue', alpha=0.5)
plt.title('Labor Hours vs Productivity')
plt.xlabel('Labor Hours')
plt.ylabel('Productivity')

# Machine Downtime vs Productivity

plt.subplot(2, 3, 2)
plt.scatter(data['Machine Downtime'], data['Productivity'], color='orange', alpha=0.5)
plt.title('Machine Downtime vs Productivity')
plt.xlabel('Machine Downtime')
plt.ylabel('Productivity')

# Energy Costs vs Cost

plt.subplot(2, 3, 3)
plt.scatter(data['Energy Costs'], data['Cost'], color='green', alpha=0.5)
plt.title('Energy Costs vs Cost')
plt.xlabel('Energy Costs')
plt.ylabel('Cost')

# Productivity vs Profit

plt.subplot(2, 3, 4)
plt.scatter(data['Productivity'], data['Profit'], color='purple', alpha=0.5)
plt.title('Productivity vs Profit')
plt.xlabel('Productivity')
plt.ylabel('Profit')

# Cost vs Profit

plt.subplot(2, 3, 5)
plt.scatter(data['Cost'], data['Profit'], color='red', alpha=0.5)
plt.title('Cost vs Profit')
plt.xlabel('Cost')
plt.ylabel('Profit')

plt.tight_layout()
plt.show()

# Histograms for Productivity, Cost, and Profit

plt.figure(figsize=(15, 5))

# Productivity histogram

plt.subplot(1, 3, 1)
plt.hist(data['Productivity'], bins=10, color='skyblue', edgecolor='black')
plt.title('Productivity Distribution')
plt.xlabel('Productivity')
plt.ylabel('Frequency')

# Cost histogram

plt.subplot(1, 3, 2)
plt.hist(data['Cost'], bins=10, color='salmon', edgecolor='black')
plt.title('Cost Distribution')
plt.xlabel('Cost')
plt.ylabel('Frequency')

# Profit histogram

plt.subplot(1, 3, 3)
plt.hist(data['Profit'], bins=10, color='lightgreen', edgecolor='black')
plt.title('Profit Distribution')
plt.xlabel('Profit')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
