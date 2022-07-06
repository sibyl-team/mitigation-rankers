import sys
import time
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
        self.contacts = None
        self.N = N
        self.T = T
        self.obs = [] 

        return True

    def _save_contacts(self, daily_contacts):
        """
        Save contacts using numpy named array
        Seems more difficult but it's not,
        and much faster than using pandas dataframes (10x)

        Assuming contacts are tuple (i,j,t,lambda) in this order
        @author Fabio Mazza
        """
        conts_dtype = np.dtype([(k, "int") for k in ["i","j","t"]]+[("lambda", "float")])

        if isinstance(daily_contacts, np.recarray):
            cts_d = daily_contacts.copy()
            cts_d.dtype.names = "i", "j", "t", "lambda"
            
        else:
            cts_d = np.array(daily_contacts,dtype=conts_dtype)

        assert len(cts_d) == len(daily_contacts)
        print(f"{len(cts_d)} new contacts,", end=" ")
        if self.contacts is None:
            self.contacts = cts_d
        else:
            self.contacts = np.concatenate((self.contacts, cts_d))
    
    def _clear_old_contacts(self, t_day:int, tau:int):
        if self.contacts is not None:
            sel = (self.contacts["t"] >= t_day - tau)
            self.contacts = self.contacts[sel]

    def rank(self, t_day, daily_contacts, daily_obs, data):
        '''
        computing rank of infected individuals
        return: list -- [(index, value), ...]
        '''
        
        for obs in daily_obs:
            self.obs.append(obs)
        
        self._clear_old_contacts(t_day, self.tau)
        #for (i,j,t,l) in daily_contacts:
        #    self.contacts.append([i,j,t,l])
        self._save_contacts(daily_contacts)

        obs_df = pd.DataFrame(self.obs, columns=["i", "s", "t_test"])
        #obs_df = obs_df[obs_df["t_test"] >= t_day - self.tau]
        contacts_df = pd.DataFrame({k: self.contacts[k] for k in ("i", "j", "t", "lambda")})
        contacts_df = contacts_df[contacts_df["t"] >= t_day - self.tau]

        print("num of contacts: ", len(contacts_df), end="; ")
        print("num of obs: ", len(obs_df), end="; ")
        
        #Score = dict([(i, 0) for i in range(self.N)])
        t0 = time.time()
        Score = pd.Series(np.zeros(self.N))
        idx_I = obs_df[obs_df["s"] == 1]["i"].to_numpy()
        idx_I_today = obs_df[(obs_df["s"] == 1) & (obs_df["t_test"] == t_day-1)]["i"].to_numpy()
        idx_dct = contacts_df[contacts_df["i"].isin(idx_I_today)]["j"].to_numpy()
        #print(len(idx_I_today), len(idx_I))
        for i in idx_I:
            Score[i] = self.N + 1e-3 * self.rng.rand()
        for i in idx_dct:
            Score[i] = 1.0 + 1e-3 * self.rng.rand()
        tend = time.time() - t0
        print(f"Done. Took {tend:4.3f} s")
        score_s = Score.sort_values(ascending=False)
        score_s = list(zip(score_s.index, score_s.values))
        #sorted_Score = list(sorted(Score.items(), key=lambda item: item[1], reverse=True))
        return score_s



