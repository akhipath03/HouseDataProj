import pandas as pd
import numpy as np

# Load original data from a CSV file
prelim_data = pd.read_csv(
    r"C:\Users\akhip\OneDrive\Desktop\HouseDataProj\originaldata.csv"
)

# Fill missing "state" values with the previous valid value using forward fill
prelim_data["state"].fillna(method="ffill", inplace=True)

# Create a DataFrame containing data only for "New Jersey"
NJ_df = prelim_data[prelim_data["state"] == "New Jersey"]

# List of column names to drop from the New Jersey DataFrame
columns_to_drop = [
    "status",
    "full_address",
    "street",
    "sold_date",
    "city",
]

# Drop specified columns from the New Jersey DataFrame
NJ_drop = NJ_df.drop(columns=columns_to_drop)

# Remove duplicated rows from the New Jersey DataFrame
deduplicated_df = NJ_drop.drop_duplicates()

# Reset the index of the DataFrame and rename the index column to "idx"
NJ_index = deduplicated_df.reset_index(drop=True)
NJ_index = NJ_index.rename_axis("idx", axis=1)


# Function to calculate the mean of a column
def calculate_mean(column):
    mean = NJ_index[column].mean()
    return mean


# Fill missing values in specific columns with calculate_mean function
NJ_index["price"].fillna(value=calculate_mean("price"), inplace=True)
NJ_index["bed"].fillna(value=round(calculate_mean("bed")), inplace=True)
NJ_index["bath"].fillna(value=round(calculate_mean("bath")), inplace=True)
NJ_index["acre_lot"].fillna(value=calculate_mean("acre_lot"), inplace=True)
NJ_index["house_size"].fillna(value=round(calculate_mean("house_size")), inplace=True)

# Create a copy of the DataFrame before outlier removal
NJ_index_copy = NJ_index.copy()


# Function to remove outliers from specified columns
def remove_outliers(df, columns, n_std):
    for col in columns:
        print("Working on column: {}".format(col))
        mean = df[col].mean()
        sd = df[col].std()
        df = df[(df[col] <= mean + (n_std * sd))]
    return df


# Remove outliers from specified columns using the remove_outliers function
removed_outliers_df = remove_outliers(
    NJ_index_copy, ["bed", "bath", "acre_lot", "price", "house_size"], 3
)

# Convert zip_code column to string and remove dots
removed_outliers_df["zip_code"] = removed_outliers_df["zip_code"].astype(str)
removed_outliers_df["zip_code"] = removed_outliers_df["zip_code"].str.replace(".", "")

# Assign the cleaned DataFrame to the variable data
data = removed_outliers_df.copy()

# Print the first few rows of the cleaned DataFrame
print(data.head())

# Save the processed data to a CSV file named processed_data.csv
data.to_csv("processed_data.csv", index=False)
