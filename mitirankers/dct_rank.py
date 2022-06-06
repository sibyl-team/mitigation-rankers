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
        #dummy obs, needed if the first time you add only one element
        #self.obs = [(0,-1,0)] 

        return True

    def rank(self, t_day, daily_contacts, daily_obs, data):
        '''
        computing rank of infected individuals
        return: list -- [(index, value), ...]
        '''

        for (i,j,t,l) in daily_contacts:
            self.contacts.append([i,j,t,l])

        obs_df = pd.DataFrame(daily_obs, columns=["i", "s", "t_test"])
        contacts_df = pd.DataFrame(self.contacts, columns=["i", "j", "t", "lambda"])
        contacts_df = contacts_df[contacts_df["t"] >= t_day - self.tau]

        Score = dict([(i, 0) for i in range(self.N)])
        idx_I = obs_df[obs_df["s"] == 1]["i"].to_numpy()
        idx_dct = contacts_df[contacts_df["i"].isin(idx_I)]["j"].to_numpy()
        for i in idx_I:
            Score[i] = N + 1e-3 * self.rng.rand()
        for i in idx_dct:
            Score[i] = 1.0 + 1e-3 * self.rng.rand()

        sorted_Score = list(sorted(Score.items(), key=lambda item: item[1], reverse=True))
        return sorted_Score



