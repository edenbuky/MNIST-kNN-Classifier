# Results & Analysis 📊

## 2. Nearest Neighbor (20 pts)
In this question, we will study the performance of the Nearest Neighbor (NN) algorithm on the MNIST dataset. The MNIST dataset consists of images of handwritten digits, along with their labels. Each image has 28 × 28 pixels, where each pixel is in gray-scale, and can get an integer value from 0 to 255. Each label is a digit between 0 and 9. The dataset has 70,000 images. Although each image is square, we treat it as a vector of size 784.

### **(b) Run the algorithm using the first n = 1000 training images, on each of the test images, using k = 10. What is the accuracy of the prediction (i.e. the percentage of correct classifications)? What would you expect from a completely random predictor?**

- As the number of samples approaches infinity, the accuracy stabilizes at **86%**.
- For a completely random classifier, since there are **10 possible digits (0-9)**, we would expect an accuracy of **10%**.

---

### **(c) Plot the prediction accuracy as a function of k, for k = 1, ..., 100 and n = 1000. Discuss the results. What is the best k?**

#### **Analysis:**
- The results indicate that the best values for `k` are **k=1** or **k=5**, as they yield the highest accuracy.
- Observing the trend, values between **k=1 and k=5** tend to provide relatively high accuracy.
- However, as `k` increases beyond a certain threshold, the accuracy declines because the classifier starts considering neighbors that are too distant and less relevant.
- Choosing `k=1` might lead to **overfitting**, as the model memorizes the training set.
- While `k=1` has slightly better accuracy than `k=5`, the difference is marginal, making `k=5` a safer choice.

#### **Graph:**
![Accuracy vs k](docs/k_vs_accuracy.png)

---

### **(d) Using k = 1, run the algorithm on an increasing number of training images. Plot the prediction accuracy as a function of n = 100, 200, ..., 5000. Discuss the results.**

#### **Analysis:**
- As seen in the graph, **increasing the training dataset size improves accuracy**, particularly in the early stages.
- After approximately **n = 1200**, the rate of accuracy improvement slows down, eventually plateauing around **90%**.

#### **Key Takeaways:**
1. **Training Data Importance:** The initial increase in accuracy demonstrates how crucial a **large training set** is for improving model performance.
2. **Learning Curve:** The model **learns quickly at first**, but eventually, additional data provides diminishing returns.
3. **Overfitting for Small n:** When `n=100`, choosing `k=1` performs poorly due to overfitting.
4. **Optimal Training Set Size:** While adding more data generally improves accuracy, **beyond a certain point (near 1200), the computational cost outweighs the performance benefits**.

#### **Graph:**
![Accuracy vs Training Size](docs/n_vs_accuracy.png)
