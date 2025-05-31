import numpy as np
import matplotlib.pyplot as plt

def parse_data(file):
    data = np.loadtxt(file)

    feature_matrix = data[:,:-1]
    labels = data[:, -1]
    return data, feature_matrix, labels

def scale_features(feature_matrix, save=None):
    means = np.mean(feature_matrix, axis=0)
    stds = np.std(feature_matrix, axis=0)
    
    scaled = (feature_matrix - means) / stds

    if save:
        np.savetxt(save, scaled)
    return scaled

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

    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h)) / m

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

def export_to_latex(filename, y_true, y_pred):
    TP, TN, FP, FN = confusion(y_true, y_pred)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{c|ccc}",
        r"\textbf{} & \textbf{} & \multicolumn{2}{c}{\textbf{Truth}} \\",
        r"\textbf{} & \textbf{} & \textbf{P} & \textbf{N} \\",
        r"\hline",
        fr"\multirow{{2}}{{*}}{{\textbf{{Prediction}}}} & \textbf{{P}} & {TP} & {FP} \\",
        fr"                                     & \textbf{{N}} & {FN} & {TN} \\",
        r"\hline",
        fr"\multicolumn{{2}}{{c|}}{{Precision}} & \multicolumn{{2}}{{c}}{{{precision:.2f}}} \\",
        fr"\multicolumn{{2}}{{c|}}{{Recall}}    & \multicolumn{{2}}{{c}}{{{recall:.2f}}} \\",
        fr"\multicolumn{{2}}{{c|}}{{F1 Score}}  & \multicolumn{{2}}{{c}}{{{F1:.2f}}} \\",
        r"\end{tabular}",
        r"\caption{Confusion matrix with precision, recall, and F1 score.}",
        r"\label{tab:confusion_metrics}",
        r"\end{table}"
    ]

    with open(filename, "w") as f:
        f.write("\n".join(lines))

def main():
    from helperscripts.optimize import quasi_newton
    import itertools

    data, M, labels = parse_data("galaxy_data.txt")
    M_scaled = scale_features(M, save="OUT/galaxy_data_scaled.txt")

    # Plot features
    plot_features(M, "figures/Q3a_unscaled")
    plot_features(M_scaled, "figures/Q3a_scaled")

    # Prepare for regression
    theta_init = np.ones(M_scaled.shape[1])

    # Wrap gradients
    f = lambda theta: cost(theta, M_scaled, labels)
    grad_f = lambda theta: cost_grad(theta, M_scaled, labels)
    
    # Find optimal theta
    theta_opt, theta_history = quasi_newton(f, grad_f, theta_init, max_iters=1000, atol=1e-8)
    
    
    ################
    ## Problem 3b ##
    ################
    # fig, ax = plt.subplots(3,2,figsize=(10,15))
    fig, ax = plt.subplots(1, 1)
    names = [r'$\kappa_{CO}$', 'Color', 'Extended', 'Emission line flux']
    plot_idx = [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]]
    for i, comb in enumerate(itertools.combinations(np.arange(0,4), 2)):
        comb = np.array(comb)
        cost_history = np.zeros(theta_history.shape[0] + 1)
        cost_history[0] = cost(theta_init, M_scaled, labels)
        for j in range(cost_history.size - 1):
            curr_theta = np.ones(4)
            # Set theta for combination to their best-fit
            curr_theta[comb] = theta_history[j, comb]
            cost_history[j + 1] = cost(curr_theta, M_scaled, labels)
        

        ax.plot(cost_history, label=f"{names[comb[0]]} + {names[comb[1]]}")
        ax.set(xlabel="Iteration", 
               ylabel=r"J($\theta$)",
               title="Cost function convergence",
               )
        ax.legend()
    plt.savefig("figures/Q3b", bbox_inches="tight", dpi=600)
    plt.close()
    
    cost_history = np.zeros(theta_history.shape[0] + 1)
    cost_history[0] = cost(theta_init, M_scaled, labels)
    for i in range(cost_history.size - 1):
        cost_history[i + 1] = cost(theta_history[i], M_scaled, labels)
    
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
    
    export_to_latex("OUT/confusion_matrix.tex", labels, predicted)
    TP, TN, FP, FN = confusion(labels, predicted)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", F1)
    
    ################
    ## Problem 3c ##
    ################

    # TODO: Determine decision boundary
    # In principle, this is at sigmoid(z) == 0.5 --> z == 0
    # and z == theta @ X, so theta @ X == 0 is the decision boundary
    # However, this is boundary is N-space. How to reduce to 2D?
    # Fix other params at what value? Project into 2D?
    fig, ax = plt.subplots(3,2,figsize=(10,15))
    names = [r'$\kappa_{CO}$', 'Color', 'Extended', 'Emission line flux']
    plot_idx = [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]]
    for i, comb in enumerate(itertools.combinations(np.arange(0,4), 2)):
        # Decision boundary is where sigmoid(z) == 0.5 --> z == 0
        # This boils down to theta @ X == 0; for our four features,
        # theta1 x1 + theta2 x2 + theta3 x3 + theta4 x4 == 0
        # The boundary is those x for which this is true; this boundary
        # is a hypersurface (4D, in this case). To plot 2D boundary, we 
        # fix the two features we're not plotting to zero, s.t. we have
        # the line theta1 x1 + theta2 x2 == 0 --> x2 = -theta1 / theta2 x1
        # where x1 is the feature plotted on the horizontal axis
        curr_horizontal_feat = M_scaled[:,comb[0]]
        boundary_x = np.linspace(curr_horizontal_feat.min(), curr_horizontal_feat.max(), 1000)
        decision_boundary = -(theta_opt[comb[0]] * boundary_x) / theta_opt[comb[1]]

        ax[plot_idx[i][0],plot_idx[i][1]].scatter(M_scaled[:,comb[0]], M_scaled[:,comb[1]], c=labels)
        ax[plot_idx[i][0],plot_idx[i][1]].set(xlabel=names[comb[0]], ylabel=names[comb[1]])
        ax[plot_idx[i][0],plot_idx[i][1]].plot(boundary_x, decision_boundary, 'k--')
    plt.savefig("figures/Q3c", bbox_inches="tight", dpi=600)
    plt.close()
if __name__ in ("__main__"):
    main()
