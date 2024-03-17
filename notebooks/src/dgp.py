"""Data-generating process class."""

import pandas as pd
from typing import List
from joblib import Parallel, delayed
from abc import abstractmethod


class DGP:

    def __init__(self,
                 n: int,
                 w: str,
                 y: List[str],
                 x: List[str] = [],
                 u: List[str] = [],
                 ):
        """Parameters for the data generating process

        Args:
            n: number of observations
            w: treatment assignment variable
            y: list of outcome names
            x: list of observable variables (confounders, features, ...) names
            u: list of unobservable variables names
        """
        self.n = n
        self.w = w
        self.y = y
        self.x = x
        self.u = u

    def __post_init__(self):
        df = self.initialize_data()
        self.df = self.add_potential_outcomes(df)

    @abstractmethod
    def initialize_data(self, seed: int = 0) -> pd.DataFrame:
        """Generates a dataframe with the baseline variables."""

    @abstractmethod
    def add_potential_outcomes(self, df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
        """Adds potential outcomes to the dataframe."""

    def check_potential_outcomes(self, df: pd.DataFrame):
        """Check that every potential outcome is in the data."""
        for y in self.y:
            for w in df[self.w].unique():
                assert f"{y}_w{w}" in df.columns

    @abstractmethod
    def add_treatment_assignment(self, df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
        """Adds the treatment assignment variable."""

    def add_realized_outcomes(self, df: pd.DataFrame, drop_unobservables: bool) -> pd.DataFrame:
        """Add realized outcomes, from potential outcomes and treatment assignment. Drop unobservables upon request."""
        for y in self.y:
            df[y] = 0
            for w in df[self.w].unique():
                df[y] += df[f"{y}_w{w}"] * (df[self.w] == w)
                if drop_unobservables:
                    del df[f"{y}_w{w}"]
        if drop_unobservables:
            for u in self.u:
                del df[u]
        return df

    def post_treatment_processing(self, df: pd.DataFrame, seed: int = 0):
        """Post-treatment processing."""
        return df

    def generate_data(self, seed_dt=0, seed_po=1, seed_as=2, seed_pt=3, drop_unobservables: bool = True, **kwargs) -> pd.DataFrame:
        """Generate potential outcomes, add assignment and select realized outcomes."""
        df = self.initialize_data(seed=seed_dt)
        df = self.add_potential_outcomes(df=df, seed=seed_po, **kwargs)
        df = self.add_treatment_assignment(df=df, seed=seed_as)
        self.check_potential_outcomes(df=df)
        df = self.add_realized_outcomes(df, drop_unobservables=drop_unobservables)
        return self.post_treatment_processing(df=df, seed=seed_pt)

    def evaluate_f_redrawing_data(self, f, n_draws: int):
        """Evaluates the function f on n_draws of the data (data, potential outcomes, and treatment assignment)."""
        results = Parallel(n_jobs=8)(delayed(f)(self.generate_data(seed_dt=i, seed_po=n_draws+1, seed_as=2*n_draws+i)) for i in range(n_draws))
        return results

    def evaluate_f_redrawing_potential_outcomes(self, f, n_draws: int):
        """Evaluates the function f on n_draws of the potential outcomes, and treatment assignment (not the data)."""
        results = Parallel(n_jobs=8)(delayed(f)(self.generate_data(seed_po=n_draws+1, seed_as=2*n_draws+i)) for i in range(n_draws))
        return results
    
    def evaluate_f_redrawing_assignment(self, f, n_draws: int):
        """Evaluates the function f on n_draws of the treatment assignment (not the data, or the potential outcomes)."""
        results = Parallel(n_jobs=8)(delayed(f)(self.generate_data(seed_as=2*n_draws+i)) for i in range(n_draws))
        return results
