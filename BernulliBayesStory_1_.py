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
import math

file_name = "clean_quercus.csv"
random_state = 42
pd.set_option('display.max_columns', None)

def bootstrap(data_size: int, boot_size: int, data, label, path_1, path_2):
    index_array = np.random.choice(data_size, boot_size - data_size)
    boot_data = data.copy().tolist()
    boot_label = label.copy().tolist()
    for i in index_array:
        # print(data[i].shape, i, boot_data.shape)
        X = boot_data[i]
        boot_data.append(X)
        # print(label[i].shape,label[i], i, boot_label.shape)
        t = boot_label[i]
        boot_label.append(t)

    boot_data = np.array(boot_data)
    boot_label = np.array(boot_label).reshape(boot_size)
    # print(boot_data.shape, boot_label.shape)
    # assert (boot_data.shape == (boot_size, data.shape[1]))
    # assert (boot_label.shape == boot_size)

    pd.DataFrame(boot_data).to_csv(path_1)
    pd.DataFrame(boot_label).to_csv(path_2)
    return boot_data, boot_label


def convolution(X, kernal_size: int):
    length = int(math.ceil(X.shape[1] / kernal_size))

    con_X = np.array([[]])
    for i in range(0, length - 1):
        select = X[:, (i * kernal_size): ((i + 1) * kernal_size)]
        if i == 0:
            con_X = np.any(select, axis=1).reshape((X.shape[0], 1)).astype(int)
        else:
            con_X = np.append(con_X, np.any(select, axis=1).reshape((X.shape[0], 1)).astype(int), axis=1)

    select = X[:, (X.shape[1] - kernal_size): X.shape[1]]
    con_X = np.append(con_X, np.any(select, axis=1).reshape((X.shape[0], 1)).astype(int), axis=1)

    assert (con_X.shape == (X.shape[0], length))
    return con_X


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
    return s[index - 1] if s[index - 1] != -1 else 0


def get_number_list_clean(s):
    """Return a clean list of numbers contained in `s`.

    Additional cleaning includes removing numbers that are not of interest
    and standardizing return list size.
    """
    s = s.replace("3-D", '')
    s = s.replace("14-dimensional", '')
    n_list = get_number_list(s)
    n_list += [-1] * (5 - len(n_list))
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


def make_bow(data, vocab):
    """
    Produce the bag-of-word representation of the data, along with a vector
    of labels. You *may* use loops to iterate over `data`. However, your code
    should not take more than O(len(data) * len(vocab)) to run.

    Parameters:
        `data`: a list of `(review, label)` pairs, like those produced from
                `list(csv.reader(open("trainvalid.csv")))`
        `vocab`: a list consisting of all unique words in the vocabulary

    Returns:
        `X`: A data matrix of bag-of-word features. This data matrix should be
             a numpy array with shape [len(data), len(vocab)].
             Moreover, `X[i,j] == 1` if the review in `data[i]` contains the
             word `vocab[j]`, and `X[i,j] == 0` otherwise.
        `t`: A numpy array of shape [len(data)], with `t[i] == 1` if
             `data[i]` is a positive review, and `t[i] == 0` otherwise.
    """
    X = np.zeros([len(data), len(vocab)])
    t = np.zeros([len(data)])

    for i in range(0, len(data)):
        if data[i][1] == '1':
            t[i] = 1
        elif data[i][1] == '2':
            t[i] = 2
        else:
            t[i] = 3
        for j in range(0, len(vocab)):
            if vocab[j] in data[i][0]:
                X[i][j] = 1
    return X, t


def process_story(df):
    # Extract story and label
    new_df = df[["q_story", "label"]]
    # print(new_df)

    data = new_df.values.tolist()
    # print(data)
    # Build the vocabulary
    vocab = set()
    for x in data:
        review = x[0]
        label = x[1]
        # print(review)
        if type(review) != str:
            review = ""
        # print(review)
        review = review.lower()
        review = review.replace(",", ' ')
        review = review.replace("?", ' ')
        review = review.replace(":", ' ')
        review = review.replace(".", ' ')
        review = review.replace("\'s", ' ')
        review = review.replace("!", ' ')
        review = review.replace("\"", ' ')
        review = review.replace("\'t", ' ')
        review = review.replace("\'", ' ')
        review = review.replace(";", ' ')

        words = review.split()
        # print(words)
        for w in words:
            vocab.add(w)
    vocab = list(vocab)
    temp = data.copy()

    for i in range(len(temp)):
        if type(temp[i][0]) != str:
            temp[i][0] = ''
    X, t = make_bow(temp, vocab)

    for i in range(len(vocab)):
        df[vocab[i]] = X[:, i]
    del df['q_story']
    return vocab


def train_model(kernal: int, x, y):

    x = convolution(x, kernal)

    n_train = 400
    n_val = 500

    y_train_abc = y[:n_train]
    y_train = y[:n_train]
    y_valid = y[n_train: n_val]
    y_test = y[n_val:]

    x_train_abc = x[:n_train]
    x_train = x[:n_train]
    x_valid = x[n_train: n_val]
    x_test = x[n_val:]


    counter = 1

    label = []

    train = []
    vali = []
    test = []

    k = 0

    for i in [[2, 3], [1, 3], [1, 2]]:
        y = df["label"]
        y1 = y.replace(i, 0)
        y1 = y1.replace(counter, 1)



        y1_train = y1[:n_train]
        y1_valid = y1[n_train: n_val]
        y1_test = y1[n_val:]

        # print("booting")
        #
        k = k + 1
        x1_train, y1_train = bootstrap(400, 400000, x_train, y1_train.values,
                                       "X_" + str(k) + ".csv", "t_" + str(k) + ".csv")

        #
        # print("finished boots")
        # Train and evaluate classifiers

        def naive_bayes_map(X, t):
            """
            Compute the parameters $pi$ and $theta_{jc}$ that maximizes the posterior
            of the provided data (X, t). We will use the beta distribution with
            $a=1$ and $b=1$ for all of our parameters.

            **Your solution should be vectorized, and contain no loops**

            Parameters:
                `X` - a matrix of bag-of-word features of shape [N, V],
                      where N is the number of data points and V is the vocabulary size.
                      X[i,j] should be either 0 or 1. Produced by the make_bow() function.
                `t` - a vector of class labels of shape [N], with t[i] being either 0 or 1.
                      Produced by the make_bow() function.

            Returns:
                `pi` - a scalar; the MAP estimate of the parameter $\pi = p(c = 1)$
                `theta` - a matrix of shape [V, 2], where `theta[j, c]` corresponds to
                          the MAP estimate of the parameter $\theta_{jc} = p(x_j = 1 | c)$
            """
            N, vocab_size = X.shape[0], X.shape[1]
            pi = 0  # TODO
            theta = np.zeros([vocab_size, 2])  # TODO

            # these matrices may be useful (but what do they represent?)
            X_positive = X[t == 1]
            X_negative = X[t == 0]

            pi = np.sum(t) / N
            theta[:, 1] = (np.sum(X_positive, axis=0) + 1) / (
                    X_positive.shape[0] + 2)
            theta[:, 0] = (np.sum(X_negative, axis=0) + 1) / (
                    X_negative.shape[0] + 2)
            # theta[:, 1] = None # you may uncomment this line if you'd like
            # theta[:, 0] = None # you may uncomment this line if you'd like

            return pi, theta


        pi_map, theta_map = naive_bayes_map(x1_train, y1_train)
        # pi_map1, theta_map1 = naive_bayes_map(x_valid, y1_valid)
        # pi_map2, theta_map2 = naive_bayes_map(x_test, y1_test)

        """## Part 3. Making predictions

        **Graded Task**: Complete the function `make_prediction` which uses our estimated parameters $\pi$ and $\theta_{jc}$ to make predictions on our dataset.

        Note that computing products of many small numbers leads to underflow. Use the fact that:

        $$a_1 \cdot a_2 \cdots a_n = e^{log(a_1) + log(a_2) + \cdots + log(a_n)} $$

        to avoid computing a product of small numbers.
        """

        def make_prediction(X, pi, theta):
            y = np.zeros(X.shape[0])
            for i in range(0, X.shape[0]):
                review = X[i]

                z_0 = 0
                for j in range(0, len(review)):
                    if int(review[j]) == 1:
                        z_0 = z_0 + np.log(theta[j][0])
                    else:
                        z_0 = z_0 + np.log((1 - theta[j][0]))

                z_1 = 0
                for j in range(0, len(review)):
                    if int(review[j]) == 1:
                        z_1 = z_1 + np.log(theta[j][1])
                    else:
                        z_1 = z_1 + np.log((1 - theta[j][1]))

                # log_a_neg = -np.log(1 - pi) - z_0
                # log_b_neg = -np.log(pi) - z_1

                # print(pi, a, b)
                # y[i] = np.exp(-log_b_neg) / np.exp(-log_a_neg + np.log(1 + np.exp(-log_b_neg + log_a_neg)))
                y[i] = pi / ((1 - pi) * np.exp(z_0 - z_1) + pi)

            return y


        def accuracy(y, t):
            return np.mean(y == t)


        train.extend([make_prediction(x_train_abc, pi_map, theta_map)])
        vali.extend([make_prediction(x_valid, pi_map, theta_map)])
        test.extend([make_prediction(x_test, pi_map, theta_map)])
        counter += 1

    train = np.array(train)
    vali = np.array(vali)
    test = np.array(test)

    y_mle_train = train.argmax(axis=0)
    y_mle_valid = vali.argmax(axis=0)
    y_mle_test = test.argmax(axis=0)

    y_mle_train += 1
    y_mle_valid += 1
    y_mle_test += 1

    print("Kernal Size:", kernal)
    print("MAP Train Acc:", accuracy(y_mle_train, y_train_abc))
    print("MAP Valid Acc:", accuracy(y_mle_valid, y_valid))
    # print("MAP test Acc:", accuracy(y_mle_test, y_test))

    return pi_map, theta_map


if __name__ == "__main__":

    df = pd.read_csv(file_name)

    # # Clean numerics
    # df["q_temperature"] = df["q_temperature"].apply(to_numeric).fillna(0)
    # median = df["q_temperature"].median()
    # df["q_temperature"] = df["q_temperature"].apply(
    #     lambda s: convert(s, median)).fillna(0)
    #
    # # Clean for number categories
    # df["q_scary"] = df["q_scary"].apply(get_number)
    # df["q_dream"] = df["q_dream"].apply(get_number)
    # df["q_desktop"] = df["q_desktop"].apply(get_number)
    # Create quote rank categories

    # df["q_quote"] = df["q_quote"].apply(get_number_list_clean)
    # temp_names = []
    #
    # for i in [1, 3]:
    #     col_name = f"picture_{i}"
    #     temp_names.append(col_name)
    #     df[col_name] = df["q_quote"].apply(lambda s: get_num(s, i))
    #
    # del df["q_quote"]
    # df["q_quote"] = df["q_quote"].apply(get_number_list_clean)
    # temp_names = []
    # for i in range(1, 6):
    #     col_name = f"rank_{i}"
    #     temp_names.append(col_name)
    #     df[col_name] = df["q_quote"].apply(lambda l: find_quote_at_rank(l, i))
    # del df["q_quote"]
    #
    # # Create category indicators
    # new_names = []
    #
    # for col in ["q_scary", "q_dream", "q_desktop"] + temp_names:
    #     indicators = pd.get_dummies(df[col], prefix=col)
    #     new_names.extend(indicators.columns)
    #     df = pd.concat([df, indicators], axis=1)
    #     del df[col]
    # # Create multi-category indicators
    #
    # for cat in ["Parents", "Siblings", "Friends", "Teacher"]:
    #     df[f"q_remind_{cat}"] = df["q_remind"].apply(lambda s: cat_in_s(s, cat))
    #     new_names.extend([f"q_remind_{cat}"])
    # del df["q_remind"]
    #
    # for cat in ["People", "Cars", "Cats", "Fireworks", "Explosion"]:
    #     df[f"q_better{cat}"] = df["q_better"].apply(lambda s: cat_in_s(s, cat))
    #     new_names.extend([f"q_better{cat}"])
    # del df["q_better"]
    # Prepare data for training - use a simple train/test split for now
    voc = process_story(df)
    df = df[voc + ["label"]]

    df = df.sample(frac=1, random_state=random_state)

    x = df.drop("label", axis=1).values
    y = df["label"].values

    for kernal_size in range(1, 11):
        train_model(kernal_size, x, y)


