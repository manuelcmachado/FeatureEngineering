
import feature_engineering_helper as feh
import pandas as pd


# --------------------------------------------------------------------------------------
#   Functions for handling missing and duplicate values
# --------------------------------------------------------------------------------------

# Get the features with nulls, the corresponding value, and percentage
def getMissingFeaturesCounts(dataFrame):
    nullCounts = dataFrame.isnull().sum()
    totals = dataFrame.isnull().count()

    nullsList = [(index, value) for index, value in nullCounts.items() if value != 0]
    totalList = [(index, value) for index, value in totals.items() if value != 0]

    missingValues = []
    for index, value in nullsList:
        for i, v in totalList:
            if index == i:
                percentage = value / v
                missingValues.append((index, value, percentage))
    return missingValues


# Get missing values in form of a data frame
def getMissingFeaturesCountsDataFrame(dataFrame):
    missingValuesDf = pd.DataFrame(data=getMissingFeaturesCounts(dataFrame),
                                   columns=['Feature', 'TotalMissingValues', 'Percentage'])
    return missingValuesDf


# Handle the missing values accordingly.
# I use the median value to replace missing Numerical features, and mode to replace the missing categorical values
def handleMissingData(dataFrame):
    featuresWithNulls = getMissingFeaturesCountsDataFrame(dataFrame)
    featuresWithNullsList = featuresWithNulls['Feature'].to_list()
    for feature in featuresWithNullsList:
        if feature in feh.getNumericFeatures(dataFrame):
            featureMedian = dataFrame[feature].median()
            dataFrame[feature].fillna(featureMedian, inplace=True)
            print(f"The '{feature}' is a numerical feature and has been replaced with '{featureMedian}'")
        elif feature in feh.getCategoricalFeatures(dataFrame):
            featureMode = dataFrame[feature].mode().values[0]
            dataFrame[feature].fillna(featureMode, inplace=True)
            print(f"The '{feature}' is a categorical feature and has been replaced with '{featureMode}'")
        elif feature in feh.getDateTimeFeatures(dataFrame):
            featureMode = dataFrame[feature].mode().values[0]
            dataFrame[feature].fillna(featureMode, inplace=True)
            print(f"The '{feature}' is a date feature and has been replaced with '{featureMode}'")
        else:
            print(f"'{feature}' has an not been handled. It has an undetermined category.")


# Remove duplicates if there are any
def removeDuplicates(dataFrame):
    if feh.checkForDuplicates(dataFrame) > 0:
        dataFrame.drop_duplicates(inplace=True)
    else:
        print('The dataset has no duplicates')


# Get a df with null columns equal or above 75 percentage
def getNullValuesAboveSeventhFivePercent(dataFrame, percentage=0.75):
    message = 'There is(are) no feature(s) with null value(s) above 75%'
    newData = pd.DataFrame(data=feh.calculateNullPercentages(dataFrame))
    newData.reset_index(inplace=True)
    percentagesAboveZero = newData.rename(columns={'index': 'Feature', 0: 'Percentage'})
    if not percentagesAboveZero[(percentagesAboveZero['Percentage'] >= percentage)].isnull:
        return message
    else:
        return percentagesAboveZero[(percentagesAboveZero['Percentage'] >= percentage)]


# Remove features with 75 or more percentage of missing values
def removeNullValuesAboveSeventhFivePercent(dataFrame):
    if type(getNullValuesAboveSeventhFivePercent(dataFrame)) == str:
        print('No feature has been removed. All are below 75%')
    else:
        for index, row in getNullValuesAboveSeventhFivePercent(dataFrame).iterrows():
            print(
                f"The column {row['Feature']} has been removed because has {row['Percentage']} percents of missing "
                f"values")
            dataFrame.drop(columns=row['Feature'], inplace=True)
