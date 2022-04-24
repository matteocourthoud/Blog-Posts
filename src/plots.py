import numpy as np
import scipy as sp
import pandas as pd
import graphviz as gr
import networkx as nx

from numpy.random import normal as rnd
from scipy.special import expit


# Plots
def dag_iv(Y="Y", T="T", Z="Z", U="U"):
    g = gr.Digraph()
    g.edge(Z, T)
    g.edge(U, Y)
    g.edge(U, T)
    g.edge(T, Y)

    c = gr.Digraph('child')
    c.attr(rank='same')
    c.node(T)
    c.node(Z)
    c.node(Y)
    g.subgraph(c)
    return g