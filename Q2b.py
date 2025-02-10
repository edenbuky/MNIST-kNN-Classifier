import Q2a as a

n_train = 1000
sample_train = a.train[:n_train]
sample_train_labels = a.train_labels[:n_train].astype(int)

n_test = 2000
sample_test = a.data[n_train:n_train + n_test]
sample_test_labels = a.labels[n_train:n_train + n_test].astype(int)


# Running the k-NN algorithm on each test image and recording accuracy
def accuracy(sample_train, sample_train_labels, sample_test, k, n_test, sample_test_labels):
    correct_predictions = 0
    for i in range(n_test):
        predicted_label = a.k_nearest_neighbor(sample_train, sample_train_labels, sample_test[i], k)
        if predicted_label == sample_test_labels[i]:
            correct_predictions += 1

    return correct_predictions / n_test


print(f"Accuracy: {accuracy(sample_train, sample_train_labels, sample_test, 10, n_test, sample_test_labels) * 100}%")
