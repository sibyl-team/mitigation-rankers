import sys
import numpy as np
import pandas as pd
from .template_rank import AbstractRanker


class DctRanker(AbstractRanker):

    def __init__(self,
                tau = 7,
                seed = 1): # time-window of seven days
        self.description = "class for naif contact tracing"
        self.tau = tau
        self.rng = np.random.RandomState(seed)

    def init(self, N, T):
        self.contacts = []
        self.N = N
        self.T = T
        self.obs = [] 

        return True

    def rank(self, t_day, daily_contacts, daily_obs, data):
        '''
        computing rank of infected individuals
        return: list -- [(index, value), ...]
        '''
        
        for obs in daily_obs:
            self.obs.append(obs)
        
        for (i,j,t,l) in daily_contacts:
            self.contacts.append([i,j,t,l])

        obs_df = pd.DataFrame(self.obs, columns=["i", "s", "t_test"])
        #obs_df = obs_df[obs_df["t_test"] >= t_day - self.tau]
        contacts_df = pd.DataFrame(self.contacts, columns=["i", "j", "t", "lambda"])
        contacts_df = contacts_df[contacts_df["t"] >= t_day - self.tau]

        print("num of contacts: ", len(contacts_df))
        print("num of observations ", len(obs_df))
        
        Score = dict([(i, 0) for i in range(self.N)])
        idx_I = obs_df[obs_df["s"] == 1]["i"].to_numpy()
        idx_I_today = obs_df[(obs_df["s"] == 1) & (obs_df["t_test"] == t_day-1)]["i"].to_numpy()
        idx_dct = contacts_df[contacts_df["i"].isin(idx_I_today)]["j"].to_numpy()
        #print(len(idx_I_today), len(idx_I))
        for i in idx_I:
            Score[i] = self.N + 1e-3 * self.rng.rand()
        for i in idx_dct:
            Score[i] = 1.0 + 1e-3 * self.rng.rand()

        sorted_Score = list(sorted(Score.items(), key=lambda item: item[1], reverse=True))
        return sorted_Score



