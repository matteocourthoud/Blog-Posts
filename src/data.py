"""
More datasets are imported from 
https://vincentarelbundock.github.io/Rdatasets/datasets.html
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def import_l86():
    """
    Lalonde (1986)
   
    Paper: https://www.jstor.org/stable/1806062
    Data: https://users.nber.org/~rdehejia/data/.nswdata2.html
    """
    # Import lalonde
    df_nsw = sm.datasets.get_rdataset('nsw_mixtape', package='causaldata').data
    df_nsw = df_nsw.drop(columns='data_id')
    df_nsw.to_csv('data/l86_nsw.csv', index=None)
    
    # Import control and merge
    df_psid = sm.datasets.get_rdataset('psid2', package='DAAG').data
    df_psid = df_psid.rename(columns={'trt': 'treat', 'nodeg': 'nodegree'})
    df_psid = pd.concat([df_nsw.loc[df_nsw['treat']==1, :], df_psid])
    df_psid.to_csv('data/l86_psid.csv', index=None)
    
    
def import_adh10():
    """
    Abadie, Diamond, Hainmueller (2010)
    
    Paper: https://www.tandfonline.com/doi/abs/10.1198/jasa.2009.ap08746
    Data: https://web.stanford.edu/~jhain/synthpage.html
    """
        
    # Import data
    df = pd.read_html('https://github.com/matheusfacure/python-causality-handbook/blob/master/causal-inference-for-the-brave-and-true/data/smoking.csv')[0]
    df = df.iloc[:, 1:]
    df = df.rename(columns={'cigsale': 'cig_sales'})
    df.loc[df['state']==1, 'state'] = 'Alabama'
    df.loc[df['state']==2, 'state'] = 'Arkansas'
    df.loc[df['state']==3, 'state'] = 'California'
    df.loc[df['state']==4, 'state'] = 'Colorado'
    df.loc[df['state']==5, 'state'] = 'Connecticut'
    df.loc[df['state']==6, 'state'] = 'Delaware'
    df.loc[df['state']==7, 'state'] = 'Georgia'
    df.loc[df['state']==8, 'state'] = 'Idaho'
    df.loc[df['state']==9, 'state'] = 'Illinois'
    df.loc[df['state']==10, 'state'] = 'Indiana'
    df.loc[df['state']==11, 'state'] = 'Iowa'
    df.loc[df['state']==12, 'state'] = 'Kansas'
    df.loc[df['state']==13, 'state'] = 'Kentucky'
    df.loc[df['state']==14, 'state'] = 'Louisiana'
    df.loc[df['state']==15, 'state'] = 'Maine'
    df.loc[df['state']==16, 'state'] = 'Minnesota'
    df.loc[df['state']==17, 'state'] = 'Mississippi'
    df.loc[df['state']==18, 'state'] = 'Missouri'
    df.loc[df['state']==19, 'state'] = 'Montana'
    df.loc[df['state']==20, 'state'] = 'Nebraska'
    df.loc[df['state']==21, 'state'] = 'Nevada'
    df.loc[df['state']==22, 'state'] = 'New Hampshire'
    df.loc[df['state']==23, 'state'] = 'New Mexico'
    df.loc[df['state']==24, 'state'] = 'North Carolina'
    df.loc[df['state']==25, 'state'] = 'North Dakota'
    df.loc[df['state']==26, 'state'] = 'Ohio'
    df.loc[df['state']==27, 'state'] = 'Oklahoma'
    df.loc[df['state']==28, 'state'] = 'Pennsylvania'
    df.loc[df['state']==29, 'state'] = 'Rhode Island'
    df.loc[df['state']==30, 'state'] = 'South Carolina'
    df.loc[df['state']==31, 'state'] = 'South Dakota'
    df.loc[df['state']==32, 'state'] = 'Tennessee'
    df.loc[df['state']==33, 'state'] = 'Texas'
    df.loc[df['state']==34, 'state'] = 'Utah'
    df.loc[df['state']==35, 'state'] = 'Vermont'
    df.loc[df['state']==36, 'state'] = 'Virginia'
    df.loc[df['state']==37, 'state'] = 'West Virginia'
    df.loc[df['state']==38, 'state'] = 'Wisconsin'
    df.loc[df['state']==39, 'state'] = 'Wyoming'
    df.to_csv('data/adh10.csv', index=None)
    
    
    
def import_l08():
    """
    Lee, Moretti, and Butler (2004)
    
    Paper: https://www.nber.org/papers/w8441
    """
    df = sm.datasets.get_rdataset('close_elections_lmb', package='causaldata').data
    df = df.dropna()
    df = df[(df['lagdemvoteshare'] < 0.75) & (df['lagdemvoteshare'] > 0.25)]
    df = df[(df['demvoteshare'] < 0.75) & (df['demvoteshare'] > 0.25)]
    df.to_csv('data/l08.csv', index=None)
    
    
def import_ck94():
    """
    Card and Krueger (1994)
    
    Paper: http://sims.princeton.edu/yftp/emet04/ck/CardKruegerMinWage.pdf
    Data: https://davidcard.berkeley.edu/data_sets.html
    """
    
    # Import data
    df = pd.read_csv("https://docs.google.com/uc?id=10h_5og14wbNHU-lapQaS1W6SBdzI7W6Z&export=download")
    
    # Clean
    df = df.dropna()
    chains = ['burgerking', 'kfc', 'roys', 'wendys']
    df['chain'] = np.select([df[f'x_{c}']==1 for c in chains], chains)
    df['id'] = df.index
    df = df[['id', 
             'chain', 
             'd_nj', 
             'y_ft_employment_before', 
             'y_ft_employment_after', 
             'x_hrs_open_weekday_before', 
             'x_hrs_open_weekday_after',
             'x_st_wage_before',
             'x_st_wage_after']]
    df.columns = ['id', 
                  'chain', 
                  'new_jersey', 
                  'employment0', 
                  'employment1', 
                  'hrsopen0', 
                  'hrsopen1',
                  'wage0', 
                  'wage1']
    df = pd.wide_to_long(df, stubnames=['employment', 'hrsopen', 'wage'], i='id', j='after').reset_index()
    
    # Save
    df.to_csv('data/ck94.csv', index=None)
    
    
    
    