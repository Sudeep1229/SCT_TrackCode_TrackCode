import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(
    "E:/Sudeep files/API_SP.POP.TOTL.MA.ZS_DS2_en_csv_v2_2287.csv",
    skiprows=4
)


# Select a continuous column (example: population percentage of males)
values = data['2020'].dropna()

# Create histogram
plt.figure()
plt.hist(values, bins=10)
plt.xlabel("Population Percentage")
plt.ylabel("Frequency")
plt.title("Distribution of Male Population Percentage (2020)")
plt.show()
