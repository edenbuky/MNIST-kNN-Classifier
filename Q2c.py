import Q2b as b
import Q2a as a
import matplotlib.pyplot as plt

n_test = 100
sample_test = a.data[b.n_train:b.n_train+n_test]
sample_test_labels = a.labels[b.n_train:b.n_train+n_test].astype(int)

accuracies = []

for k in range(1, 101):
    acc = b.accuracy(b.sample_train, b.sample_train_labels, sample_test, k, n_test, sample_test_labels)
    accuracies.append(acc)

plt.plot(range(1, 101), accuracies)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('k-NN Accuracy vs. k for n=1000')
plt.show()