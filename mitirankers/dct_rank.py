import sys
import time
import numpy as np
import pandas as pd
from .template_rank import AbstractRanker

dtype_obs = np.dtype([(k,'int') for k in ("i","s","t_test")])

class DctRanker(AbstractRanker):

    def __init__(self,
                tau = 7,
                seed = 1,
                noise=1e-3,
                debug=False): # time-window of seven days
        self.description = "class for naif contact tracing"
        self.tau = tau
        self.rng = np.random.RandomState(seed)
        self.noise=noise

        self.debug=debug
    def init(self, N, T):
        self.contacts = None
        self.N = N
        self.T = T
        self.obs = [] 

        return True

    def _save_contacts(self, daily_contacts):
        """
        Save contacts using numpy named array
        A little more complicated, but
        much faster than using pandas dataframes (10x)

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
        tstart=time.time()
        for obs in daily_obs:
            self.obs.append(tuple(obs))
        
        self._clear_old_contacts(t_day, self.tau)
        #for (i,j,t,l) in daily_contacts:
        #    self.contacts.append([i,j,t,l])
        self._save_contacts(daily_contacts)
        t_c_o = time.time()-tstart
        if self.debug: print(f"Saving cts and obs: {t_c_o:4.3f} s")
        t0 = time.time()
        obs_df = np.array(self.obs, dtype=dtype_obs)
        #obs_df=pd.DataFrame(self.obs, columns=["i", "s", "t_test"])
        #obs_df = obs_df[obs_df["t_test"] >= t_day - self.tau]
        contacts = self.contacts #pd.DataFrame({k: self.contacts[k] for k in ("i", "j", "t", "lambda")})
        contacts = contacts[contacts["t"] >= t_day - self.tau]
        t_df = time.time()-t0
        print("num of contacts: ", len(contacts), end="; ")
        print("num of obs: ", len(obs_df), end="; ")
        if self.debug: print(f"time for df: {t_df:4.3f}")
        
        noise=self.noise

        #Score = dict([(i, 0) for i in range(self.N)])
        t0 = time.time()
        Score = run_dct(self.N, contacts, obs_df, t_day, self.rng, noise)
        tend = (time.time() - t0) if self.debug else (time.time() - tstart)
        print(f"Done. Took {tend:4.3f} s")
        score_s = Score.sort_values(ascending=False)
        score_s = list(zip(score_s.index, score_s.values))
        #sorted_Score = list(sorted(Score.items(), key=lambda item: item[1], reverse=True))
        return score_s

def run_dct(N, contacts, observ, t_day, rng, noise):
    Score = pd.Series(np.zeros(N))
    idx_I = observ[observ["s"] == 1]["i"] #.to_numpy()
    idx_I_today = observ[(observ["s"] == 1) & (observ["t_test"] == t_day-1)]["i"] #.to_numpy()
    idx_dct = contacts[np.isin(contacts["i"],idx_I_today)]["j"] #.to_numpy()
    #print(len(idx_I_today), len(idx_I))
    #for i in idx_I:
    #    Score[i] = self.N + noise * self.rng.rand()
    idc = np.unique(idx_I)
    Score[idc] = N +rng.rand(len(idc))*noise
    #for i in idx_dct:
    #    Score[i] = 1.0 + noise * self.rng.rand()
    idc2 = np.unique(idx_dct)
    Score[idc2] = 1 + noise*rng.rand(len(idc2))
    return Score