import matplotlib.pyplot as plt
import Q2b as b
import Q2a as a

n_test = 1000
sample_test = a.test[:n_test]
sample_test_labels = a.test_labels[:n_test].astype(int)

accuracies = []
training_sizes = list(range(100, 5001, 100))

for n in training_sizes:
    sample_train = a.train[:n]
    sample_train_labels = a.train_labels[:n].astype(int)

    acc = b.accuracy(sample_train, sample_train_labels, sample_test, 1, n_test, sample_test_labels)
    accuracies.append(acc)

# Plot the results
plt.plot(training_sizes, accuracies)
plt.xlabel('Training Set Size (n)')
plt.ylabel('Accuracy')
plt.title('k-NN Accuracy vs. Training Set Size (n) for k=1')
plt.show()