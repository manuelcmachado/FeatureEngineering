import numpy as np
import pandas as pd
import feature_engineering as fe


# ------------------------------------------------------------------------------------------
#   Testing the feature engineering functions
# ------------------------------------------------------------------------------------------

def getData(fileLocation):
    return pd.read_csv(fileLocation, na_values=[np.nan, 'None', '?', '-999'])


def exploreDataset(fileLocation):
    dataset = getData(fileLocation)
    print(f'Dataset shape: {dataset.shape}')
    print('\n')
    print(f'Dataset columns:\n {dataset.columns}')
    print('\n')
    print('Dataset first 5 columns sample:')
    print(dataset.head())


def displayMissingData(fileLocation):
    dataset = getData(fileLocation)
    print('\n')
    print('Features with missing values:')
    print(fe.getMissingFeaturesCounts(dataset))


def displayMissingDataInDataframe(fileLocation):
    dataset = getData(fileLocation)
    percentages = fe.getMissingFeaturesCountsDataFrame(dataset)
    print('\n')
    print('Listing Features with missing values as Dataframe:')
    print(percentages)


def main():
    filePath = '/media/mmachado/LocalHDD/DataSets/Insurance/InsuranceClaimsFraud.csv'
    exploreDataset(filePath)
    displayMissingData(filePath)
    displayMissingDataInDataframe(filePath)


if __name__ == '__main__':
    main()
