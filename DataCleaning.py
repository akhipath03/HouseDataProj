import pandas as pd

# Load original data from a CSV file
prelim_df = pd.read_csv(
    r"/Users/sudeepdharanikota/Desktop/HouseDataProj/originaldata.csv"
)

# Fill missing "state" values with the previous valid value using forward fill
prelim_df["state"].fillna(method="ffill", inplace=True)

# Create a DataFrame containing data only for "New Jersey"
new_jersey_df = prelim_df[prelim_df["state"] == "New Jersey"]

# List of column names to drop from the New Jersey DataFrame
columns_to_drop = [
    "status",
    "full_address",
    "street",
    "sold_date",
    "city",
]

# Drop specified columns from the New Jersey DataFrame
dropped_columns_df = new_jersey_df.drop(columns=columns_to_drop)

# Remove duplicated rows from the New Jersey DataFrame
removed_duplicated_df = dropped_columns_df.drop_duplicates()

# Reset the index of the DataFrame and rename the index column to "idx"
indexed_df = removed_duplicated_df.reset_index(drop=True)
indexed_df = indexed_df.rename_axis("idx", axis=1)


# Function to calculate the mean of a column
def calculate_mean(column):
    mean = indexed_df[column].mean()
    return mean


# Fill missing values in specific columns with calculate_mean function
indexed_df["price"].fillna(value=calculate_mean("price"), inplace=True)
indexed_df["bed"].fillna(value=round(calculate_mean("bed")), inplace=True)
indexed_df["bath"].fillna(value=round(calculate_mean("bath")), inplace=True)
indexed_df["acre_lot"].fillna(value=calculate_mean("acre_lot"), inplace=True)
indexed_df["house_size"].fillna(value=round(calculate_mean("house_size")), inplace=True)

# Create a copy of the DataFrame before outlier removal
indexed_copy = indexed_df.copy()


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
    indexed_copy, ["bed", "bath", "acre_lot", "price", "house_size"], 3
)

# Convert zip_code column to string and remove dots
removed_outliers_df["zip_code"] = removed_outliers_df["zip_code"].astype(str)
removed_outliers_df["zip_code"] = removed_outliers_df["zip_code"].str.replace(".", "")

# Assign the cleaned DataFrame to the variable data
final_df = removed_outliers_df.copy()

# Print the first few rows of the cleaned DataFrame
print(final_df.head())

# Save the processed data to a CSV file named processed_data.csv
final_df.to_csv("processeddata.csv", index=False)
