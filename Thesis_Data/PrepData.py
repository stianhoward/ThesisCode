"""
PrepData.py

Prepare data from CompleteData.csv into 145/155 bar segregations and
randomized 80/10/10 split for training, validation, and testing sets
"""

import pandas as pd

# Train, Validation, Test data spread
DataSplit = [0.65,0.15,0.2] # If validation data isn't desired, set that term to 0

assert sum(DataSplit) == 1, "DataSplit should sum to 1"


def main():
    # Import the complete dataset
    data = pd.read_csv("completeDataTrimmed.csv", skiprows=[1])

    for bhp in data["B.H.Pressure"].unique():
        bhpData = data[data["B.H.Pressure"] == bhp]

        if DataSplit[0] != 0:
            # Training Data
            trainingData, bhpData = partition_values(bhpData, DataSplit[0])
            trainingData.to_csv(str(bhp)+"TrainingData.csv", index=False)
        if DataSplit[0] != 1:
            # There is leftover data for Validation/Testing
            if DataSplit[2] == 0:
                bhpData.to_csv(str(bhp)+"ValidationData.csv", index=False)
            elif DataSplit[1] == 0:
                bhpData.to_csv(str(bhp)+"TestData.csv", index=False)
            else:
                frac = DataSplit[1] / (DataSplit[1] + DataSplit[2])
                validationData, testData = partition_values(bhpData, frac)
                validationData.to_csv(str(bhp)+"ValidationData.csv", index=False)
                testData.to_csv(str(bhp)+"TestData.csv", index=False)



def partition_values(dataframe, fraction):
    data = dataframe.sample(frac=fraction)
    remainder = dataframe.drop(data.index)
    #return data.sample(frac=0.3), remainder
    return data, remainder


if __name__ == "__main__":
    main()
