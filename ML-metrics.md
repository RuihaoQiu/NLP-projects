## Metrics

This chapter, I am introducing the popular metrics for ML applications, as the following -

- **Classification** - confusion matrix, accuracy, precision, recall, F1-score, ROC, AUC.
- **Regression** - MSE, MAE, R squared.
- **Recommender system (learn to rank)** - AP, mAP@k, nDCG.



### Classification

- Confusion matrix
  - True positive(TP): predict positive, actual positive
  - True negative(TN): predict negative, actual negative
  - False positive(FP): predict positive, actual negative
  - False negative(FN): predict negative, actual positive

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)

- **Precision** = TP / (TP + FP)

- **Recall (True positive rate)** = TP / (TP + FN)

- **False positive rate** = FP / (FP + TN)

- **F1-score** = 2 * Precision * Recall / (Precision + Recall)

- **ROC** : x - False positive rate, y - True positive rate, threshold - 0-1

- **AUC**: Area under ROC, 1 - good, 0 - bad.



### Regression

- **MSE** = mean((y - y_pred)^2)

- **MAE** = mean(abs(y-y_pred))

- **RSME** = sqrt(MSE)

- **R_squared** = 1 - SSR/SST = 1 - sum((y - y_pred)^2)/sum((y - y_avg)^2)



### Learn to Rank (Recommender systems)

Learn to rank is to predict the rank (order) of relevant items for a given task.

- **Mean reciprocal rank (MRR)**

  Average of the reciprocal ranks of “the first relevant item” for a set of queries. **MRR** = mean(1/rank).

  

- **Precision @ k** :  

  Number of relevant items among the top k items. 

  - **P@k** = # relevant items / # top k items

  - **AP@N** = 1/n * sum(P@k)

  - **mAP@N** =  mean(AP@N)

  **Example**:

  ```markdown
  true_items = {"a", "b", "c", "d", "e", "k"}
  predict_items = ["a", "f", "d", "e", "g"]
  relevant_list = [1, 0, 1, 1, 0]
  AP@N = 1/len(true_items) * (1/1 + 0/2 + 2/3 + 3/4 + 0/5)
  or
  AP@N = 1/sum(relevant) * (1/1 + 0/2 + 2/3 + 3/4 + 0/5)
  ```



-  **Normalized Discounted Cumulative Gain (NDCG)**

  - **Cumulative Gain** :  Sum of all relevance values in a search result list, sum(rel_i)

  - **Discounted Cumulative Gain** : sum(rel_i / log2(i+1))

  **Example**:

  ```markdown
  true_items = ["a", "b", "c", "d", "e", "k"]
  relevant_scores = [6, 5, 4, 3, 2, 1]
  predict_items = ["a", "f", "d", "e", "g"]
  relevant_list = [6, 0, 3, 2, 0]
  DCG = 6/1 + 0 + 3/2 + 2/2.32 + 0
  ideal_relevant_list = [6, 3, 2, 0, 0]
  IDCG = sum(6/1 + 3/1.59 + 2/2 + 0 + 0)
  NDCG = DCG / IDCG
  ```

  

**Reference**

- https://towardsdatascience.com/20-popular-machine-learning-metrics-part-2-ranking-statistical-metrics-22c3e5a937b6

- http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html

- https://machinelearningmedium.com/2017/07/24/discounted-cumulative-gain/

- https://sigir.org/wp-content/uploads/2017/06/p243.pdf

- https://gist.github.com/tgsmith61591/d8aa96ac7c74c24b33e4b0cb967ca519