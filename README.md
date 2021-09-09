# mnist_lec
The gamma parameter defines the distance of affect of a single training example. If the data points are non-linearly separable, by applying kernel functions they are transformed so that they become linearly-separable. If the gamma values are low, the points which are few are also grouped together. If gamma values are high, the points must be very close to each other to get grouped. This might result in overfitting as they form tight bounded classes.
So from the table obtained from executing metricsplt.py

Gamma -> Accuracy -> F1 score
0.0001  ->  0.9399332591768632  -> 0.9402725064083977
0.0005  ->  0.9621802002224694  -> 0.9620809606067103
0.001  ->  0.9688542825361512  -> 0.9686644837258652
0.005  ->  0.8854282536151279  -> 0.8990436932297068
0.01  ->  0.6974416017797553  -> 0.7496380553831243
0.06  ->  0.10233592880978866  -> 0.020828108076494392
0.1  ->  0.10122358175750834  -> 0.018608779676632846
0.15  ->  0.10122358175750834  -> 0.018608779676632846
0.2  ->  0.10122358175750834  -> 0.018608779676632846

We can see that as the gamma values kept on increasing the metrics, accuracy and F1 score kept on decreasing. This is due to the overfitting caused due to very high values of gamma.

