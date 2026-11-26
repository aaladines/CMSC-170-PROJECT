# save the method charts

from . import gaussian_nb, knn, perceptron


def show_all(data, *, gnb_smoothing, knn_k, knn_p, learning_rate, max_epochs):
    gaussian_nb.show_all(data, gnb_smoothing=gnb_smoothing)
    knn.show_all(data, k=knn_k, p=knn_p)
    perceptron.show_all(data, learning_rate=learning_rate, max_epochs=max_epochs)
