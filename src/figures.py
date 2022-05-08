"""
Title:  Support Figures
Author: Matteo Courthoud
Date:   24/03/2022
"""

import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import norm

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def plot_test(mu0=0, mu1=3, sigma=1, alpha=0.05, n=100):
    """Plot statistical hypothesis test"""
    s = np.sqrt(sigma**2 / n)
    x = np.linspace(mu0 - 4*s, mu1 + 4*s, 1000)
    pdf1 = norm(mu0, s).pdf(x)
    pdf2 = norm(mu1, s).pdf(x)
    cv = mu0 + norm.ppf(1 - alpha) * s
    power = norm.cdf(np.abs(mu1 - cv) / s)

    # Plot Distributions
    plt.plot(x, pdf1, label=f'Distribution under H0: μ={mu0}');
    plt.plot(x, pdf2, label=f'Distribution under H1: μ={mu1}');

    # Plot areas
    plt.fill_between(x[x>=cv], pdf1[x>=cv], color='r', alpha=0.4, label=f'Significance: α={alpha:.2f}')
    plt.fill_between(x[x<=cv], pdf2[x<=cv], color='g', alpha=0.4, label=f'β={1-power:.2f}')

    # Vertical lines
    plt.vlines(cv, ymin=0, ymax=plt.ylim()[1], color='k', label=f'Critical Value: {cv:.2f}')
    plt.vlines(mu0, ymin=0, ymax=max(pdf1), color='k', lw=1, ls='--', label=None)
    plt.vlines(mu1, ymin=0, ymax=max(pdf2), color='k', lw=1, ls='--', label=None)

    # Other
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left");
    plt.title("Hypothesis Testing");