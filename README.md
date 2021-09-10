# mnist_lec
The code filw is assignment.py and the screenshot of the output is the image file with name 'Assignment 3 Variation of metrics with gamma'
The gamma parameter defines the distance of affect of a single training example. If the data points are non-linearly separable, by applying kernel functions they are transformed so that they become linearly-separable. If the gamma values are low, the points which are few are also grouped together. If gamma values are high, the points must be very close to each other to get grouped. This might result in overfitting as they form tight bounded classes.
So from the table obtained from assignment.py

Gamma -> Accuracy -> F1 score

1e-05  ->  0.7975528364849833  -> 0.7893123316123308

0.0001  ->  0.9399332591768632  -> 0.9402725064083977

0.0005  ->  0.9621802002224694  -> 0.9620809606067103

0.001  ->  0.9688542825361512  -> 0.9686644837258652

0.005  ->  0.8854282536151279  -> 0.8990436932297068

0.01  ->  0.6974416017797553  -> 0.7496380553831243

0.06  ->  0.10233592880978866  -> 0.020828108076494392

0.1  ->  0.10122358175750834  -> 0.018608779676632846

0.15  ->  0.10122358175750834  -> 0.018608779676632846

0.2  ->  0.10122358175750834  -> 0.018608779676632846

0.5  ->  0.10122358175750834  -> 0.018608779676632846

1  ->  0.10122358175750834  -> 0.018608779676632846

1.5  ->  0.10122358175750834  -> 0.018608779676632846

2  ->  0.10122358175750834  -> 0.018608779676632846

2.5  ->  0.10122358175750834  -> 0.018608779676632846

3  ->  0.10122358175750834  -> 0.018608779676632846

4  ->  0.10122358175750834  -> 0.018608779676632846

5  ->  0.10122358175750834  -> 0.018608779676632846

6  ->  0.10122358175750834  -> 0.018608779676632846

7  ->  0.10122358175750834  -> 0.018608779676632846

8  ->  0.10122358175750834  -> 0.018608779676632846

9  ->  0.10122358175750834  -> 0.018608779676632846

10  ->  0.10122358175750834  -> 0.018608779676632846

We can see that starting with the value of gamma as 0.00001 the metrics, accuracy and f1 score kept on increasing. They reached a peak point at 0.001. Then after 0.001 the values of metrics started to decrease and they have become stable after the gamma value of 0.06.  
The plot of these values will be like inverted parabola.
This is due to the overfitting caused due to very high values of gamma as the points close to each other get grouped and form tight bounded classes.

