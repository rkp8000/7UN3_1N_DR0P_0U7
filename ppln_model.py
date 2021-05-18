import numpy as np
from aux import get_seg


def make_b(t, t_s, t_p, t_f):
    b = np.zeros(len(t), dtype=int)

    i_s = np.zeros(t.shape)
    for start, end in t_s:
        b[(start <= t) & (t < end)] = 1
        i_s[(start <= t) & (t < end)] = 1

    i_p = np.zeros(t.shape)
    for start, end in t_p:
        b[(start <= t) & (t < end)] = 2
        i_p[(start <= t) & (t < end)] = 1

    i_f = np.zeros(t.shape)
    for start, end in t_f:
        b[(start <= t) & (t < end)] = 3
        i_f[(start <= t) & (t < end)] = 1
        
    return b, i_s, i_p, i_f


class RateBasedAdaptivePpln(object):
    """
    $$\tau_x \frac{dx}{dt} = -x + (x_sI_s + x_pI_p + x_fI_f)(1-a)$$

    $$\tau_a \frac{da}{dt} = -a + (I_s + I_p + I_f)$$

    $$0 \leq I_s + I_p + I_f \leq 1$$
    """
    
    def __init__(self, tau_x, tau_a, x_s, x_p, x_f, label=''):
        self.tau_x = tau_x if hasattr(tau_x, '__iter__') else np.array([tau_x])
        self.tau_a = tau_a if hasattr(tau_a, '__iter__') else np.array([tau_a])
        
        self.x_s = x_s if hasattr(x_s, '__iter__') else np.array([x_s])
        self.x_p = x_p if hasattr(x_p, '__iter__') else np.array([x_p])
        self.x_f = x_f if hasattr(x_f, '__iter__') else np.array([x_f])
        
        self.n = len(self.x_s)
        
        self.label = label
    
    def run_b(self, t, b, dt_bt):
        bds_s = get_seg(b==1, min_gap=0)[1]
        bds_p = get_seg(b==2, min_gap=0)[1]
        bds_f = get_seg(b==3, min_gap=0)[1]
        
        t_s = [(dt_bt*start, dt_bt*end) for start, end in bds_s]
        t_p = [(dt_bt*start, dt_bt*end) for start, end in bds_p]
        t_f = [(dt_bt*start, dt_bt*end) for start, end in bds_f]
        return self.run_b_seg(t=t, t_s=t_s, t_p=t_p, t_f=t_f)
        
    def run_b_seg(self, t, t_s, t_p, t_f):
        b, i_s, i_p, i_f = make_b(t, t_s, t_p, t_f)
        return self.run(dt=np.mean(np.diff(t)), i_s=i_s, i_p=i_p, i_f=i_f)
    
    def run(self, dt, i_s, i_p, i_f):
        
        n = self.n
        
        tau_x = self.tau_x
        tau_a = self.tau_a
        
        x_s = self.x_s
        x_p = self.x_p
        x_f = self.x_f
        
        n_t = len(i_s)
        
        x = np.nan * np.zeros((n_t, n))
        x[0, :] = 0

        a = np.nan * np.zeros((n_t, n))
        a[0, :] = 0

        for ct in range(1, n_t):
            i_total = (x_s*i_s[ct] + x_p*i_p[ct] + x_f*i_f[ct])*(1-a[ct-1])
            dx = dt/tau_x * (-x[ct-1, :] + i_total)
            x[ct, :] = x[ct-1, :] + dx

            da = dt/tau_a * (-a[ct-1, :] + i_s[ct] + i_p[ct] + i_f[ct])
            a[ct, :] = a[ct-1, :] + da
            
        return x, a


class RateBasedSSAPpln(object):
    """
    $$\tau_x\frac{dx}{dt} = -x + (x_s - a_s)I_s(t) + (x_{p} - a_{p})I_{p}(t) + (x_{f} - a_{f})I_{f}(t)$$

    $$\tau_a\frac{da_s}{dt} = -a_s + x_sI_s(t) \quad \quad \quad \textrm{sine}$$

    $$\tau_a\frac{da_{p}}{dt} = -a_{p} + x_{p}I_{p}(t) \quad \quad \quad \textrm{pulse slow}$$

    $$\tau_a\frac{da_{f}}{dt} = -a_{f} + x_{f}I_{f}(t) \quad \quad \quad \textrm{pulse fast}$$
    """
    
    def __init__(self, tau_x, tau_a, x_s, x_p, x_f, label=''):
        self.tau_x = tau_x if hasattr(tau_x, '__iter__') else np.array([tau_x])
        self.tau_a = tau_a if hasattr(tau_a, '__iter__') else np.array([tau_a])
        
        self.x_s = x_s if hasattr(x_s, '__iter__') else np.array([x_s])
        self.x_p = x_p if hasattr(x_p, '__iter__') else np.array([x_p])
        self.x_f = x_f if hasattr(x_f, '__iter__') else np.array([x_f])
        
        self.n = len(self.x_s)
        
        self.label = label
    
    def run_b(self, t, b, dt_bt):
        bds_s = get_seg(b==1, min_gap=0)[1]
        bds_p = get_seg(b==2, min_gap=0)[1]
        bds_f = get_seg(b==3, min_gap=0)[1]
        
        t_s = [(dt_bt*start, dt_bt*end) for start, end in bds_s]
        t_p = [(dt_bt*start, dt_bt*end) for start, end in bds_p]
        t_f = [(dt_bt*start, dt_bt*end) for start, end in bds_f]
        return self.run_b_seg(t=t, t_s=t_s, t_p=t_p, t_f=t_f)
       
    def run_b_seg(self, t, t_s, t_p, t_f):
        b, i_s, i_p, i_f = make_b(t, t_s, t_p, t_f)
        return self.run(dt=np.mean(np.diff(t)), i_s=i_s, i_p=i_p, i_f=i_f)
    
    def run(self, dt, i_s, i_p, i_f):
        
        n = self.n
        
        tau_x = self.tau_x
        tau_a = self.tau_a
        
        x_s = self.x_s
        x_p = self.x_p
        x_f = self.x_f
        
        n_t = len(i_s)
        
        x = np.nan * np.zeros((n_t, n))
        x[0, :] = 0

        a_s = np.nan * np.zeros((n_t, n))
        a_s[0, :] = 0

        a_p = np.nan * np.zeros((n_t, n))
        a_p[0, :] = 0

        a_f = np.nan * np.zeros((n_t, n))
        a_f[0, :] = 0

        for ct in range(1, n_t):
            i_total = (x_s - a_s[ct-1, :])*i_s[ct] + (x_p - a_p[ct-1, :])*i_p[ct] + (x_f - a_f[ct-1, :])*i_f[ct]
            dx = dt/tau_x * (-x[ct-1, :] + i_total)
            x[ct, :] = x[ct-1, :] + dx

            da_s = dt/tau_a * (-a_s[ct-1, :] + x_s*i_s[ct])
            a_s[ct, :] = a_s[ct-1, :] + da_s

            da_p = dt/tau_a * (-a_p[ct-1, :] + x_p*i_p[ct])
            a_p[ct, :] = a_p[ct-1, :] + da_p

            da_f = dt/tau_a * (-a_f[ct-1, :] + x_f*i_f[ct])
            a_f[ct, :] = a_f[ct-1, :] + da_f
            
        return x, (a_s, a_p, a_f)
