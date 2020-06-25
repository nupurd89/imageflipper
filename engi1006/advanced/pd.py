import pandas as pd

def advancedStats(data, labels):
    '''Advanced stats should leverage pandas to calculate
    some relevant statistics on the data.

    data: numpy array of data
    labels: numpy array of labels
    '''
    # convert to dataframe
    df = pd.DataFrame(data)

    # print skew and kurtosis for every column
    for i in range(len(df.columns)):
        print("Columns {} statistics".format(i))
        col = df[df.columns[i]]
        skew = col.skew()
        kurtosis = col.kurtosis()
        print("Skewness: {}\t Kurtosis: {}".format(skew, kurtosis))
        
    # assign in labels
    df["labels"] = labels

    print("\n\nDataframe statistics")

    # groupby labels into "benign" and "malignant"
    group = df.groupby("labels")

    # collect means and standard deviations for columns,
    # grouped by label
    mean = group.mean()
    stdev = group.std()

    # Print mean and stddev for Benign
    print("Benign Stats:")
    print("Mean:")
    print(mean.loc["B"])
    print("Standard Deviation:")
    print(stdev.loc["B"])

    print("\n")

    # Print mean and stddev for Malignant
    print("Malignant Stats:")
    print("Mean:")
    print(mean.loc["M"])
    print("Standard Deviation:")
    print(stdev.loc["M"])

