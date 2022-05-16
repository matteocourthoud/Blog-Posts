"""
Title:  Data Generating Processes
Author: Matteo Courthoud
Date:   24/03/2022
"""


import numpy as np
import scipy as sp
import pandas as pd

from numpy.random import normal as rnd
from scipy.special import expit
from sklearn.preprocessing import MinMaxScaler

class dgp_ad():
    """
    Data Generating Process: ads
    """
    
    def __init__(self):
        self.Y = 'revenue'
        self.T = 'ad_exposure'
        self.X = ['male', 'black', 'age', 'educ']
    
    def generate_data(self, seed=1, N=1000, oracle=False):
        np.random.seed(seed)

        # Exogenous observables
        df = pd.DataFrame({'male': np.random.binomial(1, 0.5, N),
                           'black': np.random.binomial(1, 0.5, N),
                           'age': np.rint(np.random.normal(45, 10, N))})

        # Endogenous observables
        df['educ'] = np.random.poisson(2*df['black'] + 1)

        # Treatment
        df[self.T] = (np.random.uniform(0, 2, N) \
                    + 0.3 * df['male'] \
                    - 0.2 * df['black']) > 1

        # Treatment effect
        Y0 =  0.5 * df['male'] \
            - 0.5 * df['black'] \
            + 0.1 * np.log(1 + df['educ']) \
            - 0.2 * (df['age']>50) \
            + rnd(0, 1, N)
        Y1 = Y0 \
            + 0.5 \
            + 0.4 * df['male'] \
            + 0.1 * np.log(1 + df['educ']) \
            + 0.2 * (df['age']>40)
        if oracle:
            df['Y0'] = Y0
            df['Y1'] = Y1
        df[self.Y] = Y0 + Y1 * df[self.T]
       
        return df


class dgp_aipw():
    """
    Data Generating Process for AIPW
    
    from https://www.youtube.com/watch?v=IfZHUFFlsGc
    """
    
    def __init__(self, p=20):
        self.p = p
        self.Y = 'Y'
        self.T = 'T'
        self.X = [f"x{i}" for i in range(1,p+1)]
    
    def generate_data(self, seed=1, N=1000):
        np.random.seed(seed)

        # Exogenous observables
        df = pd.DataFrame(np.random.normal(0, 1, (N, self.p)), columns=self.X)

        # Propensity score
        df['e'] = 1 / (1 + np.exp(- df['x1']))

        # Treatment
        df['T'] = (np.random.uniform(0, 1, N) < df['e']).astype(int)

        # Outcomes
        df['Y0'] = np.maximum(df['x1'] + df['x2'], 0) - 0.05 * df['T']
        df['Y1'] = np.maximum(df['x1'] + df['x3'], 0) - 0.05 * df['T'] - 0.05
        df['Y'] = df['Y0'] * (1-df['T']) + df['Y1'] * df['T']

        return df


class dgp3():
    """
    Data Generating Process 3
    """
    
    def __init__(self, p=20):
        self.p = p
        self.Y = 'Y'
        self.T = 'T'
        self.X = [f"x{i}" for i in range(1,p+1)]
    
    def generate_data(self, seed=1, N=1000):
        np.random.seed(seed)

        # Exogenous observables
        df = pd.DataFrame(np.random.normal(0, 1, (N, self.p)), columns=self.X)

        # Propensity score
        df['e'] = 1 / (1 + np.exp(- df['x1']))

        # Treatment
        df['T'] = (np.random.uniform(0, 1, N) < df['e']).astype(int)

        # Outcomes
        df['Y'] = np.maximum(df['x1'] + df['x2'] * (1-df['T']) + df['x3'] * df['T'], 0) - 0.05 * df['T']

        return df


class dgp4():
    """
    Data Generating Process 4
    
    from https://www.youtube.com/watch?v=N9ThAs7NS0g
    """
    
    def __init__(self, p=10):
        self.p = p
        self.Y = 'Y'
        self.T = 'T'
        self.X = [f"x{i}" for i in range(1,p+1)]
    
    def generate_data(self, seed=1, N=4000):
        np.random.seed(seed)

        # Exogenous observables
        df = pd.DataFrame(np.random.normal(0, 1, (N, self.p)), columns=self.X)

        # Treatment probability
        df['e'] = 0.3

        # Treatment assignment
        df['T'] = (np.random.uniform(0, 1, N) < df['e']).astype(int)
        
        # Treatment effect
        df['tau'] = 1 / (1 + np.exp(-df['x3']))

        # Outcomes
        df['Y'] = np.maximum(df['x1'] + df['x2'], 0) + df['T'] * df['tau'] + np.random.normal(0, 1, N)

        return df  


class dgp_newsletter():
    """
    Data Generating Process: reminder to sign up to newsletter
    """
    
    def __init__(self):
        self.Z = 'reminder'
        self.T = 'subscribe'
        self.Y = 'revenue'
        self.X = []
    
    def generate_data(self, seed=1, N=1000, oracle=False):
        np.random.seed(seed)
        
        # Nudge / instrument
        df = pd.DataFrame({self.Z: np.random.binomial(1, 0.5, N)})
        
        # Hidden type
        income = np.random.exponential(1, N)
        if oracle: df['income'] = income
            
        # Treatment assignment
        df[self.T] = (expit(- income + df[self.Z] + rnd(size=N)) > 0.5).astype(int)
        
        # Treatment effect
        tau = 1
        if oracle: df['tau'] = tau

        # Outcome
        df[self.Y] = -1 + 2*income + tau*df[self.T] + rnd(size=N)

        return df



class dgp_membership():
    """
    Data Generating Process: Incentives for membership
    
    inspired by: https://github.com/microsoft/EconML/blob/main/notebooks/CustomerScenarios
    """
    
    def __init__(self):
        self.Z = 'easier_signup'
        self.T = 'became_member'
        self.Y = 'revenue_post'
        self.X = []
    
    def generate_data(self, seed=1, N=100000, oracle=False):
        np.random.seed(seed)

        # Exogenous observables
        df = pd.DataFrame({'visit_flights': np.random.randint(0, 28, N),
                           'visit_hotels': np.random.randint(0, 28, N),
                           'visit_restaurants': np.random.randint(0, 28, N),
                           'visit_rental': np.random.randint(0, 28, N),
                           'origin_US': np.random.binomial(1, 0.7, N),
                           'mobile': np.random.binomial(1, 0.3, N),
                           'revenue_pre': np.random.exponential(1, N)})
        self.X = df.columns
        
        # Hidden type
        income = np.random.exponential(1, N)

        # Nudge / instrument
        df[self.Z] = np.random.binomial(1, 0.5, N)

        # Treatment probability
        e = expit(- income + df[self.Z] + rnd(size=N))
        if oracle: df['e'] = e
        
        # Treatment assignment
        df[self.T] = (np.random.uniform(0, 1, N) < e).astype(int)
        
        # Treatment effect
        tau = 0.2 + 0.3 * df['visit_flights'] - 0.2*df['visit_rental'] + df['mobile']
        if oracle: df['tau'] = tau

        # Outcome
        df[self.Y] = 1 + df['revenue_pre'] + 2*income + tau*df[self.T] + rnd(size=N)

        return df.round(2)
    

class dgp_ao18():
    """
    Data Generating Process: Anna and Olken (2018)
    
    inspired by: https://github.com/microsoft/EconML/blob/main/notebooks/CustomerScenarios
    """
    
    def __init__(self):
        self.T = 'cash_transfer'
        self.Y = 'welfare'

    def import_data(self, seed=1, oracle=False):
        np.random.seed(seed)

        # Import data
        df = pd.read_csv('data/ao18.csv')
        
        self.X = [c for c in df.columns if c not in ['hhid', 'consumption_0']]

        # Treatment 
        df['cash_transfer'] = np.random.binomial(1, 0.5, len(df))

        # Treatment effect
        df['consumption'] = df['consumption_0'] + df['cash_transfer']*100

        # Hide 
        df['welfare'] = np.log(df['consumption'])

        # Hide variables
        if oracle:
            df['welfare_0'] = np.log(df['consumption_0'])
            df['welfare_1'] = np.log(df['consumption_0'] + 100)
        else:
            df = df.drop(columns=['consumption_0'])

        # Return
        return df.dropna()
    

class dgp_did():
    """
    Data Generating Process: TV advertising
    
    Inspired by: https://matheusfacure.github.io/python-causality-handbook/13-Difference-in-Differences.html
    """
    
    def __init__(self):
        self.D = 'treated'
        self.Y = 'revenue'
        self.T = 'days'

    def generate_data(self, seed=1, N=100, T=20, oracle=False):
        np.random.seed(seed)

        # Init data
        df = pd.DataFrame(np.array(np.meshgrid(range(1, T+1), range(1,N), [0,1])).T.reshape(-1,3), 
                  columns=['day', 'id', 'treated'])
        #df['treated'] = df['treated'].astype('bool')
        
        
        # Treatment
        alpha_i = np.sqrt(df['id']) - 3*(df['id']>10)
        gamma_t = 0.1*df['day'] + rnd(size=len(df))
        tau_it = 0.5*np.log(1+df['id']) - 0.12*df['day']
        
        # Effect
        df['post'] = (df['day'] > T/2)
        df['revenue'] = alpha_i + gamma_t + \
                        1.2*df['treated'] + \
                        df['post']*df['treated']*tau_it + \
                        rnd(size=len(df))
        
        # Hide variables
        if oracle:
            df['alpha_i'] = alpha_i
            df['gamma_t'] = gamma_t
            df['tau_it'] = tau_it

        # Return
        return df
    
    
class dgp_school():
    """
    Data Generating Process: class size and test scores
    """
    
    def __init__(self):
        self.T = 'class_size'
        self.Y = 'math_score'
    
    def generate_data(self, seed=1, N=1000, oracle=False):
        np.random.seed(seed)
        
        # Dataframe
        df = pd.DataFrame({'math_hours': np.random.randint(2,5,N),
                           'history_hours': np.random.randint(2,5,N),
                           'good_school': np.random.binomial(1,0.5,N),
                           'class_year': np.random.randint(1,5,N)})
        
        # Treatment
        df[self.T] = np.random.poisson(25, N) - 1*df['class_year'] - 7*df['good_school']
        
        # Hidden ability
        ability = np.random.exponential(1, N)
        history_hours = np.random.randint(3,5,N)
            
        # Main outcome
        df[self.Y] = 1 + 0.2*df[self.T] + ability + df['math_hours'] + 5*df['good_school'] + rnd(size=N)
        
        # Other outcome
        df['hist_score'] = 1 + 0.2*df[self.T] + ability + history_hours + 5*df['good_school'] + rnd(size=N)
                
        return df
    

class dgp_marketplace():
    """
    Data Generating Process: online marketplace
    """
    
    def generate_data(self, seed=1, N=10_000):
        np.random.seed(seed)
        
        # Does the firm sells only online?
        online = np.random.binomial(1, 0.5, N)
        
        # How many products does the firm have
        products = 1 + np.random.poisson(1, N)
        
        # What is the age of the firm
        t = np.random.exponential(0.5*products, N) 
        
        # Sales
        sales = 1e3 * np.random.exponential(products + np.maximum((1 + 0.3*products + 4*online)*t - 0.5*(1 + 6*online)*t**2, 0), N)

        # Generate the dataframe
        df = pd.DataFrame({'age': t, 'sales': sales, 'online': online, 'products': products})
                
        return df
    
    
    
class dgp_store_coupons():
    """
    Data Generating Process: store coupons
    """
    
    def generate_data(self, seed=1, N=300, K=5):
        np.random.seed(seed)
        
        # Incomme
        income = np.round(np.random.normal(50, 10, N), 3) 
        
        # Using a coupon
        coupons = np.round(np.random.normal(0.5, 0.1, N) - income / 200, 3)
        
        # Day of the week
        day = np.random.choice(range(1,8), N)
        
        # Sales
        sales = np.round(10 * (income + 20*coupons + day + np.random.normal(10, 2, N)), 1)

        # Generate the dataframe
        df = pd.DataFrame({'sales': sales, 'coupons': coupons, 'income': income, 'dayofweek': [str(d) for d in day]})

        return df