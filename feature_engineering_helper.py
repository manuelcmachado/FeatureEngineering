
# --------------------------------------------------------------------------------------
#   Helper Functions
# --------------------------------------------------------------------------------------
# Get the list of features with null values. This function returns a list of the feature names only
def getFeaturesWithNulls(dataFrame):
    return [index for index in dataFrame]


# Helper function to get the numerical features.
def getNumericFeatures(dataFrame):
    return dataFrame.select_dtypes(include=['int32', 'int64', 'float32', 'float64'])


def getDateTimeFeatures(dataFrame):
    return dataFrame.select_dtypes(include=['datetime64', 'datetime32'])


# Helper function to get the categorical features
def getCategoricalFeatures(dataFrame):
    return dataFrame.select_dtypes(include=['object'])


# Helper function to calculate the percentage of each missing value
def calculateNullPercentages(dataFrame):
    total = dataFrame.isnull().count()
    nullValues = dataFrame.isnull().sum()
    return nullValues/total


# Helper function to check for duplicates in the dataset
def checkForDuplicates(dataFrame):
    return dataFrame.duplicated().sum()
