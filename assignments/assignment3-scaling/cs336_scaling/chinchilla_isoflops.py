
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from collections import defaultdict

def power_law(x, a, b):
    return a * np.power(x, b)

def main():
    with open('data/isoflops_curves.json', 'r') as f:
        data = json.load(f)

    isoflops = defaultdict(list)
    for run in data:
        isoflops[run['compute_budget']].append((run['parameters'], run['final_loss']))

    optimal_points = []
    for budget, runs in isoflops.items():
        best_run = min(runs, key=lambda x: x[1])
        optimal_points.append({'compute_budget': budget, 'parameters': best_run[0]})

    C = np.array([p['compute_budget'] for p in optimal_points])
    N_opt = np.array([p['parameters'] for p in optimal_points])
    D_opt = C / (6 * N_opt)

    # Fit model size scaling law
    popt_n, _ = curve_fit(power_law, C, N_opt)
    a_n, b_n = popt_n

    # Fit dataset size scaling law
    popt_d, _ = curve_fit(power_law, C, D_opt)
    a_d, b_d = popt_d

    # Extrapolate to new budgets
    C_extrapolate = np.logspace(23, 24, 100)
    N_extrapolate = power_law(C_extrapolate, a_n, b_n)
    D_extrapolate = power_law(C_extrapolate, a_d, b_d)

    # Plot model size scaling law
    plt.figure()
    plt.loglog(C, N_opt, 'o', label='Data')
    C_fit = np.logspace(np.log10(min(C)), 24, 100)
    N_fit = power_law(C_fit, a_n, b_n)
    plt.loglog(C_fit, N_fit, '-', label='Fit')
    plt.xlabel('Compute Budget (FLOPs)')
    plt.ylabel('Optimal Model Size (Parameters)')
    plt.title('Model Size Scaling Law')
    plt.legend()
    plt.grid(True)
    plt.savefig('model_size_scaling.png')
    print("Saved model size scaling plot to model_size_scaling.png")

    # Plot dataset size scaling law
    plt.figure()
    plt.loglog(C, D_opt, 'o', label='Data')
    D_fit = power_law(C_fit, a_d, b_d)
    plt.loglog(C_fit, D_fit, '-', label='Fit')
    plt.xlabel('Compute Budget (FLOPs)')
    plt.ylabel('Optimal Dataset Size (Tokens)')
    plt.title('Dataset Size Scaling Law')
    plt.legend()
    plt.grid(True)
    plt.savefig('dataset_size_scaling.png')
    print("Saved dataset size scaling plot to dataset_size_scaling.png")

    # Predictions
    C_23 = 1e23
    C_24 = 1e24
    N_23 = power_law(C_23, a_n, b_n)
    N_24 = power_law(C_24, a_n, b_n)
    D_23 = power_law(C_23, a_d, b_d)
    D_24 = power_law(C_24, a_d, b_d)

    print(f"Predicted optimal model size for 10^23 FLOPs: {N_23:.2e} parameters.")
    print(f"Predicted optimal model size for 10^24 FLOPs: {N_24:.2e} parameters.")
    print(f"Predicted optimal dataset size for 10^23 FLOPs: {D_23:.2e} tokens.")
    print(f"Predicted optimal dataset size for 10^24 FLOPs: {D_24:.2e} tokens.")

if __name__ == '__main__':
    main()
