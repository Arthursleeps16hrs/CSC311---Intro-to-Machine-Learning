"""
This Python file provides some useful code for reading the training file
"clean_quercus.csv". You may adapt this code as you see fit. However,
keep in mind that the code provided does only basic feature transformations
to build a rudimentary kNN model in sklearn. Not all features are considered
in this code, and you should consider those features! Use this code
where appropriate, but don't stop here!
"""

import re
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from itertools import combinations

file_name = "clean_quercus.csv"
random_state = 42
pd.set_option('display.max_columns', None)

def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """

    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)

def get_number_list(s):
    """Get a list of integers contained in string `s`
    """
    return [int(n) for n in re.findall("(\d+)", str(s))]

def get_num(s, index):
    return s[index-1] if s[index-1] != -1 else 0

def get_number_list_clean(s):
    """Return a clean list of numbers contained in `s`.

    Additional cleaning includes removing numbers that are not of interest
    and standardizing return list size.
    """
    s = s.replace("3-D", '')
    s = s.replace("14-dimensional", '')
    n_list = get_number_list(s)
    n_list += [-1]*(5-len(n_list))
    return n_list

def get_number(s):
    """Get the first number contained in string `s`.

    If `s` does not contain any numbers, return -1.
    """
    n_list = get_number_list(s)
    return n_list[0] if len(n_list) >= 1 else -1

def find_quote_at_rank(l, i):
    """Return the quote at a certain rank in list `l`.

    Quotes are indexed starting at 1 as ordered in the survey.

    If quote is not present in `l`, return -1.
    """
    return l.index(i) + 1 if i in l else -1

def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0

def convert(s, m):
    return 1 if s > m else 0

def process(df):

    # Clean numerics

    df["q_sell"] = df["q_sell"].apply(to_numeric).fillna(0)

    meds =  df["q_sell"].median()
    df["q_sell"] = df["q_sell"].apply(lambda s: convert(s, meds)).fillna(0)

    df["q_temperature"] = df["q_temperature"].apply(to_numeric).fillna(0)

    median =  df["q_temperature"].median()
    df["q_temperature"] = df["q_temperature"].apply(lambda s: convert(s, median)).fillna(0)

    # Clean for number categories
    df["q_scary"] = df["q_scary"].apply(get_number)

    df["q_dream"] = df["q_dream"].apply(get_number)

    df["q_desktop"] = df["q_desktop"].apply(get_number)

    feature["q_scary"] = ["q_scary"]
    feature["q_dream"] = ["q_dream"]
    feature["q_desktop"] = ["q_desktop"]

    # Create quote rank categories

    df["q_quote"] = df["q_quote"].apply(get_number_list_clean)

    temp_names = []

    for i in range(1,6):
        col_name = f"picture_{i}"
        temp_names.append(col_name)
        df[col_name] = df["q_quote"].apply(lambda s: get_num(s, i))
        feature[f"picture_{i}"] = [f"picture_{i}"]

    del df["q_quote"]
    # Create category indicators
    new_names = temp_names

    feature["q_scary"] = ["q_scary_-1"]
    for i in range(1, 11):
        feature["q_scary"].append(f"q_scary_{i}")

    feature["q_dream"] = ["q_dream_-1"]
    for i in range(1, 6):
        feature["q_dream"].append(f"q_dream_{i}")

    feature["q_desktop"] = ["q_desktop_-1"]
    for i in range(1, 6):
        feature["q_desktop"].append(f"q_desktop_{i}")

    for col in ["q_scary", "q_dream", "q_desktop"]:
        indicators = pd.get_dummies(df[col], prefix=col)
        new_names.extend(indicators.columns)
        df = pd.concat([df, indicators], axis=1)
        del df[col]
    # Create multi-category indicators

    feature["q_remind"] = []
    for cat in ["Parents", "Siblings", "Friends", "Teacher"]:
        df[f"q_remind_{cat}"] = df["q_remind"].apply(lambda s: cat_in_s(s, cat))
        new_names.extend([f"q_remind_{cat}"])
        feature["q_remind"].append(f"q_remind_{cat}")
    del df["q_remind"]

    feature["q_better"] = []
    for cat in ["People", "Cars", "Cats", "Fireworks", "Explosion"]:
        df[f"q_better{cat}"] = df["q_better"].apply(lambda s: cat_in_s(s, cat))
        new_names.extend([f"q_better{cat}"])
        feature["q_better"].append(f"q_better{cat}")
    del df["q_better"]

    feature['q_temperature'] = ['q_temperature']
    feature['q_sell'] = ['q_sell']

    # Prepare data for training - use a simple train/test split for now

    keys = feature.keys()

    comb = [['q_scary', 'q_dream', 'q_desktop', 'picture_1', 'picture_3', 'q_remind', 'q_better', 'q_temperature', 'q_sell']]
    df1 = df[new_names + ["q_temperature",'q_sell', "label"]]
    return comb, df1

def processing(df):
    # Clean numerics

    df["q_sell"] = df["q_sell"].apply(to_numeric).fillna(0)

    meds =  df["q_sell"].median()
    df["q_sell"] = df["q_sell"].apply(lambda s: convert(s, meds)).fillna(0)

    df["q_temperature"] = df["q_temperature"].apply(to_numeric).fillna(0)

    median =  df["q_temperature"].median()
    df["q_temperature"] = df["q_temperature"].apply(lambda s: convert(s, median)).fillna(0)

    # Clean for number categories
    df["q_scary"] = df["q_scary"].apply(get_number)

    df["q_dream"] = df["q_dream"].apply(get_number)

    df["q_desktop"] = df["q_desktop"].apply(get_number)

    feature["q_scary"] = ["q_scary"]
    feature["q_dream"] = ["q_dream"]
    feature["q_desktop"] = ["q_desktop"]

    # Create quote rank categories

    df["q_quote"] = df["q_quote"].apply(get_number_list_clean)

    temp_names = []

    for i in range(1,6):
        col_name = f"picture_{i}"
        temp_names.append(col_name)
        df[col_name] = df["q_quote"].apply(lambda s: get_num(s, i))
        feature[f"picture_{i}"] = [f"picture_{i}"]

    del df["q_quote"]
    # Create category indicators
    new_names = temp_names

    feature["q_scary"] = ["q_scary_-1"]
    for i in range(1, 11):
        feature["q_scary"].append(f"q_scary_{i}")

    feature["q_dream"] = ["q_dream_-1"]
    for i in range(1, 6):
        feature["q_dream"].append(f"q_dream_{i}")

    feature["q_desktop"] = ["q_desktop_-1"]
    for i in range(1, 6):
        feature["q_desktop"].append(f"q_desktop_{i}")

    for col in ["q_scary", "q_dream", "q_desktop"]:
        indicators = pd.get_dummies(df[col], prefix=col)
        new_names.extend(indicators.columns)
        df = pd.concat([df, indicators], axis=1)
        del df[col]
    # Create multi-category indicators

    feature["q_remind"] = []
    for cat in ["Parents", "Siblings", "Friends", "Teacher"]:
        df[f"q_remind_{cat}"] = df["q_remind"].apply(lambda s: cat_in_s(s, cat))
        new_names.extend([f"q_remind_{cat}"])
        feature["q_remind"].append(f"q_remind_{cat}")
    del df["q_remind"]

    feature["q_better"] = []
    for cat in ["People", "Cars", "Cats", "Fireworks", "Explosion"]:
        df[f"q_better{cat}"] = df["q_better"].apply(lambda s: cat_in_s(s, cat))
        new_names.extend([f"q_better{cat}"])
        feature["q_better"].append(f"q_better{cat}")
    del df["q_better"]

    feature['q_temperature'] = ['q_temperature']
    feature['q_sell'] = ['q_sell']

    # Prepare data for training - use a simple train/test split for now

    comb = [['q_scary', 'q_dream', 'q_desktop', 'picture_1', 'picture_3', 'q_remind', 'q_better', 'q_temperature', 'q_sell']]
    df1 = df[new_names + ["q_temperature"]]
    return comb, df1


if __name__ == "__main__":

    feature = {}

    df = pd.read_csv(file_name)
    df_test = pd.read_csv('example_test_set.csv')

    comb, df1 = process(df)
    _, df2 = processing(df_test)

    df2 = df2.values

    for groups in comb:
        t = []
        for f in list(groups):
           t.extend(feature[f])

        df = df1[t + ["label"]]
        df = df.sample(frac=1, random_state=random_state)

        x = df.drop("label", axis=1).values
        y = df["label"]
        n_train = 500

        x_train = x[:n_train]
        y_train = y[:n_train]

        x_test = x[n_train:]
        y_test = y[n_train:]

        # Train and evaluate classifiers
        clf = BernoulliNB()
        clf.fit(x_train, y_train)
        clf.predict(x_test)

        print(clf.predict(df2))

        train_acc = clf.score(x_train, y_train)
        test_acc = clf.score(x_test, y_test)
        print(t)
        print(f"{type(clf).__name__} train acc: {train_acc}")
        print(f"{type(clf).__name__} test acc: {test_acc}")
        print('----------------------------')
        print()
        print()

