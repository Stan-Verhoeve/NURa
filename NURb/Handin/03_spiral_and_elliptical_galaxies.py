import numpy as np
import matplotlib.pyplot as plt

def parse_data(file):
    data = np.loadtxt(file)

    feature_matrix = data[:,:-1]
    labels = data[:, -1]
    return data, feature_matrix, labels

def scale_features(feature_matrix):
    means = np.mean(feature_matrix, axis=0)
    stds = np.std(feature_matrix, axis=0)
    
    return (feature_matrix - means) / stds

def plot_features(features, savename):
    fig, ax = plt.subplots(2,2, figsize=(10,8))
    ax[0,0].hist(features[:,0], bins=20, facecolor="lightgray", edgecolor="k")
    ax[0,0].set(ylabel="N", xlabel=r"$\kappa_{CO}$")
    ax[0,1].hist(features[:,1], bins=20, facecolor="lightgray", edgecolor="k")
    ax[0,1].set(xlabel="Color")
    ax[1,0].hist(features[:,2], bins=20, facecolor="lightgray", edgecolor="k")
    ax[1,0].set(ylabel="N", xlabel="Extended")
    ax[1,1].hist(features[:,3], bins=20, facecolor="lightgray", edgecolor="k")
    ax[1,1].set(xlabel="Emission line flux")

    plt.savefig(savename, bbox_inches="tight", dpi=600)
    plt.close()

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def cost(theta, X, y):
    m = len(y)
    z = X @ theta
    # Hypothesis
    h = sigmoid(z)

    return -np.mean(y * np.log(h) + (1-y) * np.log(1-h))

def cost_grad(theta, X, y):
    m = len(y)
    z = X @ theta
    h = sigmoid(z)

    return (1. / m) * (X.T @ (h - y))

def prediction(theta, X):
    probs = sigmoid(X @ theta)
    return (probs >= 0.5).astype(int)

def confusion(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    return TP, TN, FP, FN

def main():
    from helperscripts.optimize import quasi_newton
    data, M, labels = parse_data("galaxy_data.txt")
    M_scaled = scale_features(M)
    
    # Plot features
    plot_features(M, "figures/Q3a_unscaled")
    plot_features(M_scaled, "figures/Q3a_scaled")

    # Prepare for regression
    theta_init = np.ones(M_scaled.shape[1])

    # Wrap gradients
    f = lambda theta: cost(theta, M_scaled, labels)
    grad_f = lambda theta: cost_grad(theta, M_scaled, labels)
    
    # Find optimal theta
    theta_opt, cost_history = quasi_newton(f, grad_f, theta_init, max_iters=1000, atol=1e-8)
    
    fig = plt.figure()
    ax =fig.add_subplot(111)
    ax.plot(cost_history)
    ax.set(xlabel="Iteration",
           ylabel=r"J($\theta$)",
           title="Cost function convergence",
           )
    fig.savefig("figures/Q3b_full_history", bbox_inches="tight", dpi=600)
    plt.close()

    # Get model predictions
    predicted = prediction(theta_opt, M_scaled)

    TP, TN, FP, FN = confusion(labels, predicted)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", F1)
if __name__ in ("__main__"):
    main()
