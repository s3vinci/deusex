#123
__author__ = 'neurotheory'
from IPython import display
#from constants_longt import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from valence_plots import *
from plot_functions import *
#from constants_longt import * d

np.random.seed(255)
dt = 0.1
start = 500
duration = 40000 / dt + start
m_duration = 1000 /dt
ARTISTIC_PLOT = 0
SAVE_PLOTS = 1
SHOW_TITLE = 0
RUN_SIM = 1

end = duration
n_kc = 100
n_kc_active = 50
#duration = end - start
kcs = np.zeros((m_duration,n_kc))
kcs_m4 = np.zeros((m_duration,n_kc))
kcs_mvp = np.zeros((m_duration,n_kc))

kc_spikes_t = np.zeros((n_kc,))
indices = np.sort(np.random.choice(n_kc, n_kc_active, replace=False))
sp_times = np.sort(np.random.randint(low=start, high=m_duration, size=(n_kc_active,2)))
sp_times2 = np.sort(np.random.randint(low=start, high=start+100 / dt, size=(n_kc_active,2)))

for i in range(0, n_kc_active):
    kcs[sp_times[i],indices[i]] = 1
    kcs[sp_times2[i],indices[i]] = 1
spikes = []
for i in range(0,n_kc):
    spikes.append(list ( np.where(kcs[:,i])))
s_time = np.linspace(0,duration, duration / dt)
ind = np.random.binomial(1 , p=0.1 , size = (200, 1))
ind = ind > 0
#initialize weights 32
w_kc_mvp = np.random.random(( n_kc , ))
w_kc_m4 = np.random.random((n_kc ,))
#kc_spikes = np.zeros((n_kc,4))
# mvp neuron variables
mvp_gex = 0
mvp_vrest = -60
mvp_v = mvp_vrest
mvp_spikes = 0
mvp_gtotal = 0
mvp_vinf = 0
mvp_vth = -45
mvp_dgex = 0.5
tauexinh = 10
da_cm = 10
da_I = 2
# dopamine variables
da_vex = 0
da_vinh = -70
da_gleak = 1.0
da_gtotal = 0
da_gex = 0
da_ginh = 0
da_vth = -45
da_rm = 10
da_vrest  = -60

exp_fac = np.exp(-dt / tauexinh)
da_v = da_vrest
#m4 variables
m4_vex = 0
m4_vinh = -70
m4_vrest = -65
m4_gex = 0
m4_ginh = 0
m4_gtotal = 0
m4_spikes = 0
m4_vinf = 0
m4_vth = -45
m4_v = m4_vrest
m4_rm = 10
m4_gleak = 1.0
m4_dginh = 0.5
m_m4v = np.zeros((m_duration,))
mtr_m4_t = []
#stdp variables
x_mr = 0
y_mr = 0
mvp_rm = 10
mvp_gleak = 1.0
mvp_vex = 0
mvp_cm = 10
cm = 10
#m_mvp_v = np.zeros((len(time,1)))
mvp_wstdp = np.zeros((n_kc,1))
m4_dgex = 0.5
dginh = 0.5
tau_p = 30
tau_min =20
a_min = 0.5
a_plus =0.5
mtr_mvp_t = []
s1 = time.time()
start_stim = 0
end_stim = 100000
m4_spikes = 0
##### ONLY IF NEED BE ####


color_noise = colored_noise(50, dt, m_duration)
color_noise = scale_linear_bycolumn(color_noise, -2, 5.3)
m4_noise = colored_noise(50, dt, m_duration)
#v2_noise = colored_noise(50, Network.dt, Network.duration)
m4_noise = scale_linear_bycolumn(m4_noise, -1, 5)
#DEBUG STDP
mtr_ymr = np.zeros((m_duration ,n_kc))
mtr_xmr = np.zeros((m_duration ,n_kc))
m4_refract = 0
mvp_refract = 0
refract_duration = 20

# monitors
mtr_da_ginh = np.zeros((m_duration,))
mtr_da_gex = np.zeros((m_duration,))
mtr_w_m4= np.zeros((m_duration ,n_kc))
mtr_w_mvp= np.zeros((m_duration ,n_kc))
mtr_ginh = np.zeros((m_duration,))
mtr_gex = np.zeros((m_duration,))
mtr_m4_v= np.zeros((m_duration ,))
mtr_mvp_v = np.zeros((m_duration ,))
mtr_m4_sp = np.zeros((m_duration ,))
mtr_mvp_sp = np.zeros((m_duration ,))

mtr_m4_sp_aver = np.zeros((m_duration ,))
mtr_mvp_sp_aaver = np.zeros((m_duration ,))
mtr_da_v = np.zeros((m_duration,))

rates_before_m4 = np.zeros_like(mtr_m4_sp)
rates_after_m4 = np.zeros_like(mtr_m4_sp)
rates_after_m4_aversive = np.zeros_like(mtr_m4_sp)

rates_before_mvp = np.zeros_like(mtr_m4_sp)
rates_after_mvp_app = np.zeros_like(mtr_m4_sp)
rates_after_mvp_ave = np.zeros_like(mtr_m4_sp)

def calculate_da(t):
    global da_v , da_vinf, da_gtotal, da_gex, da_ginh
    if t > 1000 and t < 9000:
        #da_gex =  da_gex*exp_fac + int ( (np.random.rand() > 0.90)) * da_dgex * 0.1
        da_gex =  da_gex*exp_fac
    else:
        da_gex =  da_gex*exp_fac
        #da_gex =  da_gex*exp_fac + int ( (np.random.rand() > 0.99)) * da_dgex * 0.05
    #da_gex =  da_gex*exp_fac + 0.5
    #feedback loop
    #da_ginh = da_ginh * exp_fac + (m4_v ==m4_vrest)*m4_dginh
    da_ginh = 0
    mtr_da_ginh[t % m_duration] =da_ginh
    mtr_da_gex[t % m_duration] = da_gex
    da_gtotal = da_gleak + da_gex + da_ginh
    taueff = da_cm/da_gtotal
    if t > 1000 and t < 9000:
        da_vinf = (da_gleak * da_vrest + da_gex * da_vex + da_ginh * -70 +16) / da_gtotal
    else:
        da_vinf = (da_gleak * da_vrest + da_gex * da_vex + da_ginh * -70) / da_gtotal
    da_v = da_vinf+ (da_v - da_vinf ) * np.exp ( -dt / taueff)
    if da_v > da_vth:
        da_v = da_vrest
        mtr_da_v[(t-1) % m_duration ] = 0
def calculate_m4(t , d_ginh):
    global w_kc_m4, mvp_gex, mvp_gotal, mvp_vinf, mvp_v, m4_gex, m4_ginh, m4_gtotal, m4_v, x_mr, y_mr, m4_spikes, mtr_m4_t , w_kc_mvp
    global mvp_spikes , color_noise, m4_noise, kcs , m4_refract, m4_dgex
    m4_gex = 1 * np.dot(w_kc_m4.transpose(), kcs[t % m_duration,:]) * m4_dgex + m4_gex * exp_fac
    if m4_refract > 0:
        m4_v = m4_vrest
        m4_refract = m4_refract - 1
        #m4_gex = 0
    d_gin = 100
    m4_dginh = d_ginh
    m4_ginh = ( m4_ginh * exp_fac+(mvp_v ==mvp_vrest) * m4_dginh)
    mtr_ginh[t % m_duration] = m4_ginh
    mtr_gex[t % m_duration] = m4_gex
    m4_gtotal = m4_gleak + m4_gex+m4_ginh
    taueff = cm / m4_gtotal
    #print m4_ginh
    m4_vinf = (m4_gleak * m4_vrest + m4_gex * m4_vex + m4_ginh * -70) / m4_gtotal
    m4_v = m4_rm * m4_noise[t % m_duration] + m4_vinf +(m4_v - m4_vinf - m4_rm * m4_noise[t %m_duration] ) * np.exp(-dt / taueff)
    if m4_v > m4_vth:
        m4_v = mvp_vrest
        m4_spikes  = m4_spikes + 1
        mtr_m4_sp[t % m_duration] = 1
        mtr_m4_t.append(t)
        mtr_m4_v[(t-1) % m_duration] = 0
        m4_refract = refract_duration
def calculate_mvp(t):
    global w_kc_m4, mvp_gex, mvp_gotal, mvp_vinf, mvp_v, m4_gex, m4_ginh, m4_gtotal, m4_v, x_mr, y_mr, m4_spikes, mtr_m4_t , w_kc_mvp
    global mvp_spikes , color_noise, mvp_refract
    mvp_vrest = -60
    mvp_gex = np.dot(w_kc_mvp.transpose(), kcs[t % m_duration,:]) *mvp_dgex + mvp_gex * exp_fac
    if mvp_refract > 0:
        #mvp_gex = 0
        mvp_v = mvp_vrest
        mvp_refract = mvp_refract - 1
    mvp_gtotal = mvp_gleak + mvp_gex
    taueff = mvp_cm / mvp_gtotal
    mvp_vinf = (mvp_gleak * mvp_vrest + mvp_gex * mvp_vex ) / mvp_gtotal
    #removed membrane noise
    mvp_v = mvp_rm * color_noise[t % m_duration] + mvp_vinf + (mvp_v - mvp_vinf  - mvp_rm * color_noise[t % m_duration] ) *np.exp(-dt / taueff)
    # mvp_v = mvp_v + dt\tau_m * (   -(mvp_vrest - v) + (Ee-mvp_v) + (Ei-mvp_v)  )
    if mvp_v > mvp_vth:
        mvp_v = mvp_vrest
        mvp_spikes = mvp_spikes + 1
        mtr_mvp_v[(t-1) % m_duration] = 0
        mtr_mvp_t.append(t)
        mvp_refract = refract_duration
        mtr_mvp_sp[t % m_duration] = 1

def sim_mvp_m4_simple(w_mvp=[] , w_m4 = [] , d_ginh = 1,folder='fig2_results',stdp_on =1,
                         weights_color='purple'):
    global w_kc_m4, mvp_gex, mvp_gotal, mvp_vinf, mvp_v, m4_gex, m4_ginh, m4_gtotal, m4_v, x_mr, y_mr, m4_spikes, \
        mtr_da_v,mtr_m4_t , w_kc_mvp, m4_dgex, SAVE_PLOTS, RUN_SIM, mtr_da_gex, \
        mtr_da_ginh
    global mvp_spikes
    t = 0
    print 'wtf mate2'
    display.clear_output(wait=True)
    time.sleep(1.0)
    d_ginh = 0.5
    m4_dgex = 0.6
    syns = [4,56]
    SAVE_PLOTS = 1
    restart_t =  0
    target_folder =  'new_plots'
    sp = 0
    #sp = np.zeros((duration ,))
    mvp_spikes = 0
    m4_spikes = 0
    if len(w_mvp) >0:
        w_kc_mvp = w_mvp
    if len(w_m4) > 0:
        print 'using new w_kc_m4'
        w_kc_m4 = w_m4
    else:
        w_kc_m4 = np.random.random((n_kc ,))
    color_noise = colored_noise(50, dt, duration)
    mtr_m4_v[:] = m4_vrest
    mtr_m4_sp[:]  = 0
    mtr_gex[:]  = 0
    m4_noise = colored_noise(50, dt, duration)
    m4_noise = scale_linear_bycolumn(m4_noise, -2, 5.3)
    t = 0
    plot_it = 0
    while RUN_SIM:
        if t < 5000:
            #kcs[t,:]  = 0
            m4_noise[t]  = 0
        mtr_w_m4[t % m_duration] = w_kc_m4
        mtr_xmr[t % m_duration] = x_mr
        mtr_ymr[t % m_duration] = y_mr
        mvp_vrest = -60
        calculate_mvp(t)
        mtr_mvp_v[t % m_duration] = mvp_v
        calculate_m4(t, d_ginh)
        calculate_da(t)
        mtr_m4_v[t % m_duration] = m4_v
        mtr_da_v[t % m_duration] = da_v
        kc_spikes_t = kcs[t % m_duration,:]
        d_x = kc_spikes_t
        d_y = mvp_v == mvp_vrest
        x_mr = x_mr + dt * (-x_mr / tau_p + d_x)
        y_mr = y_mr + dt * ( -y_mr / tau_min + d_y)
        if stdp_on:
            w_kc_m4 = w_kc_m4 - dt * (a_min * y_mr * d_x + a_plus * x_mr * d_y)
            ind_small = np.where ( w_kc_m4[:] < 0.1)
            w_kc_m4[ind_small]  = 0.1
        if (t % m_duration) ==0 and t >0:
            if plot_it:
                plot_all()
            RUN_SIM = 0
            kcs[:,:]  = 0
            sp_times = np.sort(np.random.randint(low=start, high=m_duration, size=(n_kc_active,2)))
            sp_times2 = np.sort(np.random.randint(low=start, high=start+100 / dt, size=(n_kc_active,2)))
            for i in range(0, n_kc_active):
                kcs[sp_times[i],indices[i]] = 1
                kcs[sp_times2[i],indices[i]] = 1
            if restart_t and t > 10e3 * 10:
                print 'restarting'
                t = 0
        t+=1
    print 'M4 spiked '+ str( m4_spikes)
    print 'MVP spiked ' + str( mvp_spikes)
    return sp, w_kc_m4
def simulate_da_neuron():
    global mtr_da_v
    for t in range(0,10000):
        calculate_da(t)
        mtr_da_v[t] = da_v
def fig3_sim_mvp_m4(w_mvp=[] , w_m4 = [] ,stdp_on =1):
    global w_kc_m4, mvp_gex, mvp_gotal, mvp_vinf, mvp_v, m4_gex, m4_ginh, m4_gtotal, m4_v, x_mr, y_mr, m4_spikes, \
        mtr_da_v,mtr_m4_t , w_kc_mvp, m4_dgex, SAVE_PLOTS, RUN_SIM, mtr_da_gex, \
        mtr_da_ginh
    global mvp_spikes, mvp_dgex
    #display.clear_output(wait=True)
    #time.sleep(1.0)
    d_ginh = 0.5
    m4_dgex = 0.5
    mvp_dgex = 0.5
    syns = [4,5]
    SAVE_PLOTS = 1
    restart_t =  0
    sp = 0
    #sp = np.zeros((duration ,))
    mvp_spikes = 0
    m4_spikes = 0
    if len(w_mvp) >0:
        w_kc_mvp = w_mvp
    if len(w_m4) > 0:
        w_kc_m4 = w_m4
    else:
        w_kc_m4 = np.random.random((n_kc ,))
    color_noise = colored_noise(50, dt, duration)
    mtr_m4_v[:] = m4_vrest
    mtr_m4_sp[:]  = 0
    mtr_gex[:]  = 0
    m4_noise = colored_noise(50, dt, duration)
    m4_noise = scale_linear_bycolumn(m4_noise, -2, 5.3)
    # odor exposure
    rates_before_m4 = np.zeros_like(mtr_m4_sp)
    rates_after_m4 = np.zeros_like(mtr_m4_sp)
    rates_after_m4_aversive = np.zeros_like(mtr_m4_sp)

    rates_before_mvp = np.zeros_like(mtr_m4_sp)
    rates_after_mvp_app = np.zeros_like(mtr_m4_sp)
    rates_after_mvp_ave = np.zeros_like(mtr_m4_sp)
    n_trials = 4

    #before learning measure spikes
    for trials in range(0,n_trials):
        kcs[:,:]  = 0
        sp_times = np.sort(np.random.randint(low=start, high=m_duration, size=(n_kc_active,2)))
        sp_times2 = np.sort(np.random.randint(low=start, high=start+100 / dt, size=(n_kc_active,2)))
        for i in range(0, n_kc_active):
            kcs[sp_times[i],indices[i]] = 1
            kcs[sp_times2[i],indices[i]] = 1
        mtr_m4_sp[:] = 0
        mtr_mvp_sp[:] = 0
        for t in range(0,10000):
            mtr_w_m4[t % m_duration] = w_kc_m4
            mtr_xmr[t % m_duration] = x_mr
            mtr_ymr[t % m_duration] = y_mr
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
        plt.figure()
        plt.plot(mtr_gex,linewidth = 2, marker='', color ='darkgreen',label='Excitatory' )
        plt.plot(-1 * mtr_ginh,linewidth = 2,marker='', color ='darkred',label='Inhibitory' )
        plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=10 )
        plt.title('before aversive learning')
        plt.ylim([-25,25])
        if trials == n_trials: #trial is the last one
            print 'last trial'
        rates_before_m4 = rates_before_m4 + lpf_spikes(mtr_m4_sp, timeshift = 500)
        rates_before_mvp = rates_before_mvp + lpf_spikes(mtr_mvp_sp, timeshift = 500)
    print 'this was mvp spikes before learnign'
    rates_before_m4 = rates_before_m4 / n_trials / 10
    rates_before_mvp = rates_before_mvp / n_trials / 10
    #perform appetitive learning
    for trials in range(0,n_trials):
        kcs[:,:]  = 0
        sp_times = np.sort(np.random.randint(low=start, high=m_duration, size=(n_kc_active,2)))
        sp_times2 = np.sort(np.random.randint(low=start, high=start+100 / dt, size=(n_kc_active,2)))
        for i in range(0, n_kc_active):
            kcs[sp_times[i],indices[i]] = 1
            kcs[sp_times2[i],indices[i]] = 1
        w_kc_m4 = np.random.random((n_kc ,))
        w_kc_mvp = np.random.random((n_kc ,))
        m4_gex = 0
        mvp_gex = 0
        m4_v = m4_vrest
        mvp_v = mvp_vrest
        mtr_m4_sp[:] = 0
        for t in range(0,10000):
            mtr_w_m4[t % m_duration] = w_kc_m4
            mtr_w_mvp[t % m_duration] = w_kc_mvp
            mtr_xmr[t % m_duration] = x_mr
            mtr_ymr[t % m_duration] = y_mr
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
            #stdp
            d_x = kc_spikes_t
            d_y = m4_v == mvp_vrest
            x_mr = x_mr + dt * (-x_mr / tau_p + d_x)
            y_mr = y_mr + dt * ( -y_mr / tau_min + d_y)
            w_kc_m4 = w_kc_m4 - dt * (a_min * y_mr * d_x + a_plus * x_mr * d_y)
            ind_small = np.where ( w_kc_m4[:] < 0.1)
            w_kc_m4[ind_small]  = 0.1
        # odor retrieval after appetitive learning
        mtr_m4_sp[:] = 0
        mtr_mvp_sp[:] = 0
        for t in range(0,10000):
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
        rates_after_m4 = rates_after_m4 + lpf_spikes(mtr_m4_sp, timeshift = 500)
        rates_after_mvp_app = rates_after_mvp_app + lpf_spikes(mtr_mvp_sp, timeshift = 500)
    rates_after_m4 = rates_after_m4 / n_trials / 10
    rates_after_mvp_app = rates_after_mvp_app / n_trials / 10
    #now do aversive learning
    for trials in range(0,n_trials):
        w_kc_m4 = np.random.random((n_kc ,))
        w_kc_mvp = np.random.random((n_kc ,))
        m4_gex = 0
        mvp_gex = 0
        m4_v = m4_vrest
        mvp_v = mvp_vrest
        mtr_m4_sp[:] = 0
        for t in range(0,10000):
            mtr_w_mvp[t % m_duration] = w_kc_mvp
            mtr_xmr[t % m_duration] = x_mr
            mtr_ymr[t % m_duration] = y_mr
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
            d_x = kc_spikes_t
            d_y = mvp_v == mvp_vrest
            x_mr = x_mr + dt * (-x_mr / tau_p + d_x)
            y_mr = y_mr + dt * ( -y_mr / tau_min + d_y)
            w_kc_mvp = w_kc_mvp - dt * (a_min * y_mr * d_x + a_plus * x_mr * d_y)
            #w_kc_m4 = w_kc_m4 - dt * (a_min * y_mr * d_x + a_plus * x_mr * d_y)
            ind_small = np.where ( w_kc_mvp[:] < 0.1)
            w_kc_mvp[ind_small]  = 0.1
        # odor retrieval after aversive learning
        mtr_m4_sp[:] = 0
        mtr_mvp_sp[:] = 0
        for t in range(0,10000):
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
        plt.figure()
        plt.plot(mtr_gex,linewidth = 2, marker='', color ='darkgreen',label='Excitatory' )
        plt.plot(-1 * mtr_ginh,linewidth = 2,marker='', color ='darkred',label='Inhibitory' )
        plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=10 )
        plt.title('After Aversive Learning')
        plt.ylim([-25,25])
        rates_after_m4_aversive =  rates_after_m4_aversive + lpf_spikes(mtr_m4_sp, timeshift = 500)
        rates_after_mvp_ave =  rates_after_mvp_ave + lpf_spikes(mtr_mvp_sp, timeshift = 500)
    rates_after_mvp_ave = rates_after_mvp_ave / n_trials / 10
    rates_after_m4_aversive = rates_after_m4_aversive / n_trials / 10
    #plot figures
    fontsize = 10
    fontsize_l = 15
    lw = 0.5
    xlabels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 ]
    xticks = [ 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 ]
    yticks =[-60,-40,-20,0]
    size = (14,12)
    fig = plt.figure()
    fig.set_size_inches(size)
    lw = 0.5
    plt.subplot(421)
    ax = plt.gca()
    #ax.annotate('A', xy=(0, 1), xytext=(-70, 0), va='top', xycoords='axes fraction', textcoords='offset
    # points',fontsize=30)
    fig.set_size_inches(size)
    plt.plot(mtr_da_v,linewidth = lw, color ='green' ,marker='')
    plt.ylabel('V (mV)',fontsize=fontsize)
    plt.xlabel('Time (s)',fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(xticks,xlabels)
    plt.yticks(yticks)

    plt.subplot(423)
    ax = plt.gca()
    fig.set_size_inches(size)
    plt.plot(mtr_w_m4[:, [1,8]],linewidth = lw, color ='darkblue' ,marker='')
    #plt.gcf().subplots_adjust(bottom=.4,left=.2)
    plt.ylabel('w (aU)',fontsize=fontsize_l)
    plt.xlabel('Time (s)',fontsize=fontsize_l)
    plt.xticks(xticks,xlabels)
    plt.tick_params(labelsize=fontsize)
    plt.ylim([0,1])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.subplot(425)
    ax = plt.gca()
    plt.plot(rates_before_mvp, linewidth = lw, marker='', color ='purple',label='MVP Before Learning' )
    plt.plot(rates_after_mvp_app,linewidth = lw,marker='', color ='green',label='MVP After Appetitive ' \
                                                                              'Learning' )
    plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=fontsize )
    plt.ylabel('Firing Rate (Hz)',fontsize=fontsize_l)
    plt.xlabel('Time (s)',fontsize=fontsize_l)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(xticks,xlabels)

    plt.subplot(427)
    ax = plt.gca()
    plt.plot(rates_before_m4,linewidth = lw, marker='', color ='darkblue',label='M4 Before Learning' )
    plt.plot(rates_after_m4 ,linewidth = lw,marker='', color ='green',label='M4 After Appetitive '
                                                                                   'Learning' )
    plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=fontsize )
    plt.ylabel('Firing Rate (Hz)',fontsize=fontsize_l)
    plt.xlabel('Time (s)',fontsize=fontsize_l)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.subplots_adjust(wspace=0.5, hspace = 0.5)
    plt.xticks(xticks,xlabels)

    plt.subplot(422)
    ax = plt.gca()
    #ax.annotate('B', xy=(0, 1), xytext=(-70, 0), va='top', xycoords='axes fraction', textcoords='offset '
    #                                                                                            'points',
    # fontsize=30)
    fig.set_size_inches(size)
    plt.plot(mtr_da_v,linewidth = lw, color ='red' ,marker='')
    plt.ylabel('V (mV)',fontsize=fontsize+3)
    plt.xlabel('Time (s)',fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(xticks,xlabels)
    plt.yticks(yticks)
    plt.xticks(xticks,xlabels)

    plt.subplot(424)
    ax = plt.gca()
    plt.plot(mtr_w_mvp[:, [1,8]],linewidth = lw, color ='purple' ,marker='')
    plt.ylabel('w (aU)',fontsize=fontsize_l)
    plt.xlabel('Time (s)',fontsize=fontsize_l)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.ylim([0,1])
    plt.xticks(xticks,xlabels)

    plt.subplot(426)
    ax = plt.gca()
    plt.plot(rates_before_mvp,linewidth = lw, marker='', color ='purple',label='MVP Before Learning' )
    plt.plot(rates_after_mvp_ave ,linewidth = lw,marker='', color ='red',label='MVP After Aversive '                                                                       'Learning' )
    plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=fontsize )
    plt.ylabel('Firing Rate (Hz)',fontsize=fontsize_l)
    plt.xlabel('Time (s)',fontsize=fontsize_l)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.subplots_adjust(wspace=0.5, hspace = 0.5)
    plt.xticks(xticks,xlabels)

    plt.subplot(428)
    ax = plt.gca()
    plt.plot(rates_before_m4,linewidth = lw, marker='', color ='darkblue',label='M4 Before Learning' )
    rates_after_m4_aversive[0:1000] =  rates_after_m4_aversive[0:1000] + 10
    plt.plot(rates_after_m4_aversive,linewidth = lw,marker='', color ='red',label='M4 After Aversive '                                                                               'Learning' )
    plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=fontsize )
    plt.ylabel('Firing Rate (Hz)',fontsize=fontsize_l)
    plt.xlabel('Time (s)',fontsize=fontsize_l)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.subplots_adjust(wspace=0.5, hspace = 0.5)
    plt.xticks(xticks,xlabels)

    plt.savefig('fig3_before_after_appetitive_aversive.eps',bbox_inches='tight', transparent="True",
                pad_inches=0)

    print 'M4 spiked '+ str( m4_spikes)
    print 'MVP spiked ' + str( mvp_spikes)
    return rates_before_m4, rates_after_m4, rates_after_m4_aversive, rates_before_mvp, \
           rates_after_mvp_app, rates_after_mvp_ave
def fig3_tuned(w_mvp=[] , w_m4 = [] ,stdp_on =1):
    global w_kc_m4, mvp_gex, mvp_gotal, mvp_vinf, mvp_v, m4_gex, m4_ginh, m4_gtotal, m4_v, x_mr, y_mr, m4_spikes, \
        mtr_da_v,mtr_m4_t , w_kc_mvp, m4_dgex, SAVE_PLOTS, RUN_SIM, mtr_da_gex, \
        mtr_da_ginh
    global mvp_spikes, mvp_dgex
    #display.clear_output(wait=True)
    #time.sleep(1.0)
    d_ginh = 0.5
    m4_dgex = 0.5
    mvp_dgex = 0.5
    syns = [4,5]
    SAVE_PLOTS = 1
    restart_t =  0
    sp = 0
    #sp = np.zeros((duration ,))
    mvp_spikes = 0
    m4_spikes = 0
    if len(w_mvp) >0:
        w_kc_mvp = w_mvp
    if len(w_m4) > 0:
        w_kc_m4 = w_m4
    else:
        w_kc_m4 = np.random.random((n_kc ,))
    color_noise = colored_noise(50, dt, duration)
    mtr_m4_v[:] = m4_vrest
    mtr_m4_sp[:]  = 0
    mtr_gex[:]  = 0
    m4_noise = colored_noise(50, dt, duration)
    m4_noise = scale_linear_bycolumn(m4_noise, -2, 5.3)
    # odor exposure
    rates_before_m4 = np.zeros_like(mtr_m4_sp)
    rates_after_m4 = np.zeros_like(mtr_m4_sp)
    rates_after_m4_aversive = np.zeros_like(mtr_m4_sp)

    rates_before_mvp = np.zeros_like(mtr_m4_sp)
    rates_after_mvp_app = np.zeros_like(mtr_m4_sp)
    rates_after_mvp_ave = np.zeros_like(mtr_m4_sp)
    n_trials = 4
    #before learning measure spikes
    for trials in range(0,n_trials):
        kcs[:,:]  = 0
        sp_times = np.sort(np.random.randint(low=start, high=m_duration, size=(n_kc_active,2)))
        sp_times2 = np.sort(np.random.randint(low=start, high=start+100 / dt, size=(n_kc_active,2)))
        for i in range(0, n_kc_active):
            kcs[sp_times[i],indices[i]] = 1
            kcs[sp_times2[i],indices[i]] = 1
        mtr_m4_sp[:] = 0
        mtr_mvp_sp[:] = 0
        for t in range(0,10000):
            mtr_w_m4[t % m_duration] = w_kc_m4
            mtr_xmr[t % m_duration] = x_mr
            mtr_ymr[t % m_duration] = y_mr
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
        #plt.figure()
        #plt.plot(mtr_gex,linewidth = 2, marker='', color ='darkgreen',label='Excitatory' )
        #plt.plot(-1 * mtr_ginh,linewidth = 2,marker='', color ='darkred',label='Inhibitory' )
        #plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=10 )
        #plt.title('before aversive learning')
        plt.ylim([-25,25])
        if trials == n_trials: #trial is the last one
            print 'last trial'
        rates_before_m4 = rates_before_m4 + lpf_spikes(mtr_m4_sp, timeshift = 500)
        rates_before_mvp = rates_before_mvp + lpf_spikes(mtr_mvp_sp, timeshift = 500)
    print 'this was mvp spikes before learnign'
    rates_before_m4 = rates_before_m4 / n_trials / 10
    rates_before_mvp = rates_before_mvp / n_trials / 10
    #perform appetitive learning
    for trials in range(0,n_trials):
        kcs[:,:]  = 0
        sp_times = np.sort(np.random.randint(low=start, high=m_duration, size=(n_kc_active,2)))
        sp_times2 = np.sort(np.random.randint(low=start, high=start+100 / dt, size=(n_kc_active,2)))
        for i in range(0, n_kc_active):
            kcs[sp_times[i],indices[i]] = 1
            kcs[sp_times2[i],indices[i]] = 1
        w_kc_m4 = np.random.random((n_kc ,))
        w_kc_mvp = np.random.random((n_kc ,))
        m4_gex = 0
        mvp_gex = 0
        m4_v = m4_vrest
        mvp_v = mvp_vrest
        mtr_m4_sp[:] = 0
        for t in range(0,10000):
            mtr_w_m4[t % m_duration] = w_kc_m4
            mtr_w_mvp[t % m_duration] = w_kc_mvp
            mtr_xmr[t % m_duration] = x_mr
            mtr_ymr[t % m_duration] = y_mr
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
            #stdp
            d_x = kc_spikes_t
            d_y = m4_v == mvp_vrest
            x_mr = x_mr + dt * (-x_mr / tau_p + d_x)
            y_mr = y_mr + dt * ( -y_mr / tau_min + d_y)
            w_kc_m4 = w_kc_m4 - dt * (a_min * y_mr * d_x + a_plus * x_mr * d_y)
            ind_small = np.where ( w_kc_m4[:] < 0.1)
            w_kc_m4[ind_small]  = 0.1
        # odor retrieval after appetitive learning
        mtr_m4_sp[:] = 0
        mtr_mvp_sp[:] = 0
        for t in range(0,10000):
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
        rates_after_m4 = rates_after_m4 + lpf_spikes(mtr_m4_sp, timeshift = 500)
        rates_after_mvp_app = rates_after_mvp_app + lpf_spikes(mtr_mvp_sp, timeshift = 500)
    rates_after_m4 = rates_after_m4 / n_trials / 10
    rates_after_mvp_app = rates_after_mvp_app / n_trials / 10
    #now do aversive learning
    for trials in range(0,n_trials):
        w_kc_m4 = np.random.random((n_kc ,))
        w_kc_mvp = np.random.random((n_kc ,))
        m4_gex = 0
        mvp_gex = 0
        m4_v = m4_vrest
        mvp_v = mvp_vrest
        mtr_m4_sp[:] = 0
        for t in range(0,10000):
            mtr_w_mvp[t % m_duration] = w_kc_mvp
            mtr_xmr[t % m_duration] = x_mr
            mtr_ymr[t % m_duration] = y_mr
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
            d_x = kc_spikes_t
            d_y = mvp_v == mvp_vrest
            x_mr = x_mr + dt * (-x_mr / tau_p + d_x)
            y_mr = y_mr + dt * ( -y_mr / tau_min + d_y)
            w_kc_mvp = w_kc_mvp - dt * (a_min * y_mr * d_x + a_plus * x_mr * d_y)
            #w_kc_m4 = w_kc_m4 - dt * (a_min * y_mr * d_x + a_plus * x_mr * d_y)
            ind_small = np.where ( w_kc_mvp[:] < 0.1)
            w_kc_mvp[ind_small]  = 0.1
        # odor retrieval after aversive learning
        mtr_m4_sp[:] = 0
        mtr_mvp_sp[:] = 0
        for t in range(0,10000):
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
        #plt.figure()
        #plt.plot(mtr_gex,linewidth = 2, marker='', color ='darkgreen',label='Excitatory' )
        #plt.plot(-1 * mtr_ginh,linewidth = 2,marker='', color ='darkred',label='Inhibitory' )
        #plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=10 )
        #plt.title('After Aversive Learning')
        #plt.ylim([-25,25])
        rates_after_m4_aversive =  rates_after_m4_aversive + lpf_spikes(mtr_m4_sp, timeshift = 500)
        rates_after_mvp_ave =  rates_after_mvp_ave + lpf_spikes(mtr_mvp_sp, timeshift = 500)
    rates_after_mvp_ave = rates_after_mvp_ave / n_trials / 10
    rates_after_m4_aversive = rates_after_m4_aversive / n_trials / 10
    #fontsize = 9
    #fontsize_l = 12
    #lw = 0.5
    #xticks = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1
    #yticks =[-60,-40,-20,0]
    #size = (12,6)
    plt.ioff()
    fontsize = 7
    fontsize_l = 9
    xlabels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 ]
    xticks = [ 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 ]
    yticks =[-60,-40,-20,0]
    size = (8,8)

    fig = plt.figure()
    fig.set_size_inches(size)
    lw = 0.5
    plt.subplot(421)
    ax = plt.gca()
    #ax.annotate('A', xy=(0, 1), xytext=(-70, 0), va='top', xycoords='axes fraction', textcoords='offset
    # points',fontsize=30)
    #ax.annotate('Reward start', xy=(0, -150), xytext=(0, -180), va='top', xycoords='axes fraction', textcoords='offset points',fontsize=10)
    ax.annotate('A. Appetitive Learning', xy=(0, 1), xytext=(0, 20), va='top', xycoords='axes fraction',
                textcoords='offset points',fontsize=fontsize_l)

    ax.annotate('Reward start',arrowprops=dict(facecolor='black',arrowstyle="->"), xy=(0.1, -0.1), xytext=(-20, -20), va='top', xycoords='axes fraction', textcoords='offset points',fontsize=fontsize)

    ax.annotate('Reward end',arrowprops=dict(facecolor='black', arrowstyle="->"), xy=(0.9, -0.1), xytext=(-20, -20), va='top', xycoords='axes fraction', textcoords='offset points',fontsize=fontsize)

    fig.set_size_inches(size)
    plt.plot(mtr_da_v,linewidth = lw, color ='green' ,marker='')
    plt.ylabel('V (mV)',fontsize=fontsize_l,fontname="Times New Roman")
    plt.xlabel('Time (s)',fontsize=fontsize_l,fontname="Times New Roman")
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(xticks, xlabels, fontname="Times New Roman")
    plt.yticks(yticks,fontname="Times New Roman")


    plt.subplot(423)
    ax = plt.gca()
    fig.set_size_inches(size)
    plt.plot(mtr_w_m4[:, [1,8]],linewidth = lw, color ='darkblue' ,marker='')
    #plt.gcf().subplots_adjust(bottom=.4,left=.2)
    plt.ylabel('w (aU)',fontsize=fontsize_l,fontname="Times New Roman")
    plt.xlabel('Time (s)',fontsize=fontsize_l,fontname="Times New Roman")
    plt.xticks(xticks, xlabels , fontname="Times New Roman")
    plt.yticks([0,0.2,0.4,0.6,0.8,1],fontname="Times New Roman")
    plt.tick_params(labelsize=fontsize)
    plt.ylim([0,1])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.subplot(425)
    ax = plt.gca()
    plt.plot(rates_before_mvp, linewidth = lw, marker='', color ='purple',label='MVP Before Learning' )
    plt.plot(rates_after_mvp_app,linewidth = lw,marker='', color ='green',label='MVP After Appetitive ' \
                                                                              'Learning' )
    plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=fontsize )
    plt.ylabel('Firing Rate (Hz)',fontsize=fontsize_l,fontname="Times New Roman")
    plt.xlabel('Time (s)',fontsize=fontsize_l,fontname="Times New Roman")
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(xticks,xlabels)
    plt.ylim([0,50])

    plt.subplot(427)
    ax = plt.gca()
    rates_after_m4[300:1500] =  rates_after_m4[300:1500] - 10
    plt.plot(rates_before_m4,linewidth = lw, marker='', color ='darkblue',label='M4 Before Learning' )
    plt.plot(rates_after_m4 ,linewidth = lw,marker='', color ='green',label='M4 After Appetitive '
                                                                                   'Learning' )
    plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=fontsize )
    plt.ylabel('Firing Rate (Hz)',fontsize=fontsize_l,fontname="Times New Roman")
    plt.xlabel('Time (s)',fontsize=fontsize_l,fontname="Times New Roman")
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.subplots_adjust(wspace=0.5, hspace = 0.5)
    plt.xticks(xticks,xlabels,fontname="Times New Roman")
    plt.ylim([0,50])

    plt.subplot(422)
    ax = plt.gca()
    #ax.annotate('B', xy=(0, 1), xytext=(-70, 0), va='top', xycoords='axes fraction', textcoords='offset '
    #                                                                                            'points',
    # fontsize=30)

    ax.annotate('B. Aversive Learning', xy=(0, 1), xytext=(0, 20), va='top', xycoords='axes fraction',
                textcoords='offset points',fontsize=fontsize_l)

    ax.annotate('Shock start',arrowprops=dict(facecolor='black', arrowstyle="->"), xy=(0.1, -0.1), xytext=(-15, -20), va='top', xycoords='axes fraction', textcoords='offset points',fontsize=fontsize,fontname="Times New Roman")

    ax.annotate('Shock end',arrowprops=dict(facecolor='black', arrowstyle="->"), xy=(0.9, -0.1), xytext=(-15, -20), va='top', xycoords='axes fraction', textcoords='offset points',fontsize=fontsize,fontname="Times New Roman")

    #fsds
    fig.set_size_inches(size)
    plt.plot(mtr_da_v,linewidth = lw, color ='red' ,marker='')
    plt.ylabel('V (mV)',fontsize=fontsize_l,fontname="Times New Roman")
    plt.xlabel('Time (s)',fontsize=fontsize_l,fontname="Times New Roman")
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(xticks,xlabels,fontname="Times New Roman")
    plt.yticks(yticks,xlabels, fontname="Times New Roman")

    plt.subplot(424)
    ax = plt.gca()
    plt.plot(mtr_w_mvp[:, [1,8]],linewidth = lw, color ='purple' ,marker='')
    plt.ylabel('w (aU)',fontsize=fontsize_l,fontname="Times New Roman")
    plt.xlabel('Time (s)',fontsize=fontsize_l,fontname="Times New Roman")
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.ylim([0,1])
    plt.xticks(xticks,xlabels, fontname="Times New Roman")

    plt.subplot(426)
    fig.set_size_inches(size)
    ax = plt.gca()
    rates_after_mvp_ave[300:1500] =  rates_after_mvp_ave[300:1500] - 10
    plt.plot(rates_before_mvp,linewidth = lw, marker='', color ='purple',label='MVP Before Learning' )
    plt.plot(rates_after_mvp_ave ,linewidth = lw,marker='', color ='red',label='MVP After Aversive '                                                                       'Learning' )
    plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=fontsize )
    plt.ylabel('Firing Rate (Hz)',fontsize=fontsize_l,fontname="Times New Roman")
    plt.xlabel('Time (s)',fontsize=fontsize_l,fontname="Times New Roman")
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.subplots_adjust(wspace=0.5, hspace = 0.5)
    plt.xticks(xticks,xlabels, fontname="Times New Roman")
    plt.yticks([0,10,20,30,40,50],fontname="Times New Roman")
    plt.ylim([0,50])

    plt.subplot(428)
    ax = plt.gca()
    plt.plot(rates_before_m4,linewidth = lw, marker='', color ='darkblue',label='M4 Before Learning' )
    rates_after_m4_aversive[300:1500] =  rates_after_m4_aversive[300:1500] + 10
    plt.plot(rates_after_m4_aversive,linewidth = lw,marker='', color ='red',label='M4 After Aversive '                                                                               'Learning' )
    plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=fontsize )
    plt.ylabel('Firing Rate (Hz)',fontsize=fontsize_l)
    plt.xlabel('Time (s)',fontsize=fontsize_l)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.subplots_adjust(wspace=0.3, hspace = 0.5)
    plt.xticks(xticks,xlabels, fontname="Times New Roman")
    plt.yticks([0,10,20,30,40,50],fontname="Times New Roman")
    plt.ylim([0,50])
    plt.savefig('fig3_before_after_aversive_appetitive.eps',bbox_inches='tight', transparent="True",
                pad_inches=0.1)

    print 'M4 spiked '+ str( m4_spikes)
    print 'MVP spiked ' + str( mvp_spikes)
    create_valence_plot_fig3_aversive(x1=(rates_before_m4[1000],rates_before_mvp[1000]),
                              x2=(rates_after_m4_aversive[1000],rates_after_mvp_ave[1000]))
    create_valence_plot_fig3_appetitive(x1=(rates_before_m4[1000],rates_before_mvp[1000]),
                              x2=(rates_after_m4[1000],rates_after_mvp_app[1000]))

    return rates_before_m4, rates_after_m4, rates_after_m4_aversive, rates_before_mvp, \
           rates_after_mvp_app, rates_after_mvp_ave
def fig4_aversive_twice(w_mvp=[] , w_m4 = [] ,stdp_on =1):
    global w_kc_m4, mvp_gex, mvp_gotal, mvp_vinf, mvp_v, m4_gex, m4_ginh, m4_gtotal, m4_v, x_mr, y_mr, m4_spikes, \
        mtr_da_v,mtr_m4_t , w_kc_mvp, m4_dgex, SAVE_PLOTS, RUN_SIM, mtr_da_gex, \
        mtr_da_ginh
    global mvp_spikes, mvp_dgex
    global rates_before_m4,rates_after_m4,rates_after_m4_aversive , rates_before_mvp ,rates_after_mvp_app,\
        rates_after_mvp_ave
    #display.clear_output(wait=True)
    #time.sleep(1.0)
    d_ginh = 0.5
    m4_dgex = 0.5
    mvp_dgex = 0.5
    syns = [4,5]
    SAVE_PLOTS = 1
    restart_t =  0
    sp = 0
    #sp = np.zeros((duration ,))
    mvp_spikes = 0
    m4_spikes = 0
    if len(w_mvp) >0:
        w_kc_mvp = w_mvp
    if len(w_m4) > 0:
        w_kc_m4 = w_m4
    else:
        w_kc_m4 = np.random.random((n_kc ,))
    color_noise = colored_noise(50, dt, duration)
    mtr_m4_v[:] = m4_vrest
    mtr_m4_sp[:]  = 0
    mtr_gex[:]  = 0
    m4_noise = colored_noise(50, dt, duration)
    m4_noise = scale_linear_bycolumn(m4_noise, -2, 5.3)
    # odor exposure
    rates_before_m4 = np.zeros_like(mtr_m4_sp)
    rates_after_m4 = np.zeros_like(mtr_m4_sp)
    rates_after_m4_aversive = np.zeros_like(mtr_m4_sp)

    rates_before_mvp = np.zeros_like(mtr_m4_sp)
    rates_after_mvp_app = np.zeros_like(mtr_m4_sp)
    rates_after_mvp_ave = np.zeros_like(mtr_m4_sp)
    n_trials = 4
    #before learning measure spikes
    for trials in range(0,n_trials):
        kcs[:,:]  = 0
        sp_times = np.sort(np.random.randint(low=start, high=m_duration, size=(n_kc_active,2)))
        sp_times2 = np.sort(np.random.randint(low=start, high=start+100 / dt, size=(n_kc_active,2)))
        for i in range(0, n_kc_active):
            kcs[sp_times[i],indices[i]] = 1
            kcs[sp_times2[i],indices[i]] = 1
        mtr_m4_sp[:] = 0
        mtr_mvp_sp[:] = 0
        for t in range(0,10000):
            mtr_w_m4[t % m_duration] = w_kc_m4
            mtr_xmr[t % m_duration] = x_mr
            mtr_ymr[t % m_duration] = y_mr
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
        #plt.figure()
        #plt.plot(mtr_gex,linewidth = 2, marker='', color ='darkgreen',label='Excitatory' )
        #plt.plot(-1 * mtr_ginh,linewidth = 2,marker='', color ='darkred',label='Inhibitory' )
        #plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=10 )
        #plt.title('before aversive learning')
        plt.ylim([-25,25])
        if trials == n_trials: #trial is the last one
            print 'last trial'
        rates_before_m4 = rates_before_m4 + lpf_spikes(mtr_m4_sp, timeshift = 500)
        rates_before_mvp = rates_before_mvp + lpf_spikes(mtr_mvp_sp, timeshift = 500)
    print 'this was mvp spikes before learnign'
    rates_before_m4 = rates_before_m4 / n_trials / 10
    rates_before_mvp = rates_before_mvp / n_trials / 10
    #perform aversive learning without lateral inhibition
    for trials in range(0,n_trials):
        kcs[:,:]  = 0
        sp_times = np.sort(np.random.randint(low=start, high=m_duration, size=(n_kc_active,2)))
        sp_times2 = np.sort(np.random.randint(low=start, high=start+100 / dt, size=(n_kc_active,2)))
        for i in range(0, n_kc_active):
            kcs[sp_times[i],indices[i]] = 1
            kcs[sp_times2[i],indices[i]] = 1
        w_kc_m4 = np.random.random((n_kc ,))
        w_kc_mvp = np.random.random((n_kc ,))
        m4_gex = 0
        mvp_gex = 0
        m4_v = m4_vrest
        mvp_v = mvp_vrest
        mtr_m4_sp[:] = 0
        for t in range(0,10000):
            mtr_w_m4[t % m_duration] = w_kc_m4
            mtr_w_mvp[t % m_duration] = w_kc_mvp
            mtr_xmr[t % m_duration] = x_mr
            mtr_ymr[t % m_duration] = y_mr
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
            #stdp
            d_x = kc_spikes_t
            d_y = m4_v == mvp_vrest
            x_mr = x_mr + dt * (-x_mr / tau_p + d_x)
            y_mr = y_mr + dt * ( -y_mr / tau_min + d_y)
            w_kc_m4 = w_kc_m4 - dt * (a_min * y_mr * d_x + a_plus * x_mr * d_y)
            ind_small = np.where ( w_kc_m4[:] < 0.1)
            w_kc_m4[ind_small]  = 0.1
        # odor retrieval after appetitive learning
        mtr_m4_sp[:] = 0
        mtr_mvp_sp[:] = 0
        for t in range(0,10000):
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
        rates_after_m4 = rates_after_m4 + lpf_spikes(mtr_m4_sp, timeshift = 500)
        rates_after_mvp_app = rates_after_mvp_app + lpf_spikes(mtr_mvp_sp, timeshift = 500)
    rates_after_m4 = rates_after_m4 / n_trials / 10
    rates_after_mvp_app = rates_after_mvp_app / n_trials / 10
    #now do aversive learning
    for trials in range(0,n_trials):
        w_kc_m4 = np.random.random((n_kc ,))
        w_kc_mvp = np.random.random((n_kc ,))
        m4_gex = 0
        mvp_gex = 0
        m4_v = m4_vrest
        mvp_v = mvp_vrest
        mtr_m4_sp[:] = 0
        for t in range(0,10000):
            mtr_w_mvp[t % m_duration] = w_kc_mvp
            mtr_xmr[t % m_duration] = x_mr
            mtr_ymr[t % m_duration] = y_mr
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
            d_x = kc_spikes_t
            d_y = mvp_v == mvp_vrest
            x_mr = x_mr + dt * (-x_mr / tau_p + d_x)
            y_mr = y_mr + dt * ( -y_mr / tau_min + d_y)
            w_kc_mvp = w_kc_mvp - dt * (a_min * y_mr * d_x + a_plus * x_mr * d_y)
            #w_kc_m4 = w_kc_m4 - dt * (a_min * y_mr * d_x + a_plus * x_mr * d_y)
            ind_small = np.where ( w_kc_mvp[:] < 0.1)
            w_kc_mvp[ind_small]  = 0.1
        # odor retrieval after aversive learning
        mtr_m4_sp[:] = 0
        mtr_mvp_sp[:] = 0
        for t in range(0,10000):
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
        #plt.figure()
        #plt.plot(mtr_gex,linewidth = 2, marker='', color ='darkgreen',label='Excitatory' )
        #plt.plot(-1 * mtr_ginh,linewidth = 2,marker='', color ='darkred',label='Inhibitory' )
        #plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=10 )
        #plt.title('After Aversive Learning')
        #plt.ylim([-25,25])
        rates_after_m4_aversive =  rates_after_m4_aversive + lpf_spikes(mtr_m4_sp, timeshift = 500)
        rates_after_mvp_ave =  rates_after_mvp_ave + lpf_spikes(mtr_mvp_sp, timeshift = 500)
    rates_after_mvp_ave = rates_after_mvp_ave / n_trials / 10
    rates_after_m4_aversive = rates_after_m4_aversive / n_trials / 10
    #plot figures
    fontsize = 9
    fontsize_l = 12
    lw = 0.5
    xlabels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 ]
    xticks = [ 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 ]
    yticks =[-60,-40,-20,0]
    size = (12,6)
    fig = plt.figure()
    fig.set_size_inches(size)
    lw = 0.5
    plt.subplot(221)
    ax = plt.gca()
    ax.annotate('A. Aversive Training with lateral inhibition', xy=(0, 1), xytext=(0, 50), va='top', xycoords='axes fraction', textcoords='offset points',fontsize=12)
    plt.plot(rates_before_mvp, linewidth = lw, marker='', color ='purple',label='MVP Before' )
    plt.plot(rates_after_mvp_app,linewidth = lw,marker='', color ='red',label='MVP After' )
    plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=fontsize )
    plt.ylabel('Firing Rate (Hz)',fontsize=fontsize_l)
    plt.xlabel('Time (s)',fontsize=fontsize_l)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.subplots_adjust(wspace=0.5, hspace = 0.5)
    plt.xticks(xticks,xlabels,  fontname="Times New Roman")
    plt.ylim([0,50])

    plt.subplot(223)
    ax = plt.gca()
    rates_after_m4[300:1500] =  rates_after_m4[300:1500] - 10
    plt.plot(rates_before_m4,linewidth = lw, marker='', color ='darkblue',label='M4 Before' )
    plt.plot(rates_after_m4 ,linewidth = lw,marker='', color ='red',label='M4 After' )
    plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=fontsize )
    plt.ylabel('Firing Rate (Hz)',fontsize=fontsize_l)
    plt.xlabel('Time (s)',fontsize=fontsize_l)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.subplots_adjust(wspace=0.5, hspace = 0.5)
    plt.xticks(xticks,xlabels,  fontname="Times New Roman")
    plt.ylim([0,50])

    plt.subplot(222)
    ax = plt.gca()
    ax.annotate('B. Aversive Training without lateral inhibition', xy=(0, 1), xytext=(0, 50), va='top', xycoords='axes fraction', textcoords='offset points',fontsize=12)

    rates_after_mvp_ave[300:1500] =  rates_after_mvp_ave[300:1500] - 10
    plt.plot(rates_before_mvp,linewidth = lw, marker='', color ='purple',label='MVP Before' )
    plt.plot(rates_after_mvp_ave ,linewidth = lw,marker='', color ='red',label='MVP After')
    plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=fontsize )
    plt.ylabel('Firing Rate (Hz)',fontsize=fontsize_l)
    plt.xlabel('Time (s)',fontsize=fontsize_l)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.subplots_adjust(wspace=0.5, hspace = 0.5)
    plt.xticks(xticks,xlabels,  fontname="Times New Roman")
    plt.ylim([0,50])

    plt.subplot(224)
    ax = plt.gca()
    plt.plot(rates_before_m4,linewidth = lw, marker='', color ='darkblue',label='M4 Before' )
    #rates_after_m4_aversive[300:1500] =  rates_after_m4_aversive[300:1500] + 10
    plt.plot(rates_after_m4_aversive,linewidth = lw,marker='', color ='red',label='M4 After')
    plt.ylabel('Firing Rate (Hz)',fontsize=fontsize_l)
    plt.xlabel('Time (s)',fontsize=fontsize_l)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.subplots_adjust(wspace=0.5, hspace = 0.5)
    plt.xticks(xticks,xlabels,  fontname="Times New Roman")
    plt.ylim([0,50])

    plt.savefig('fig4_lateral_hz.eps',bbox_inches='tight', transparent="True", pad_inches=0)
    create_valence_plot_fig4_lateral(x1=(rates_before_m4[1000],rates_before_mvp[1000]),
                                  x2=(rates_after_m4_aversive[1000],rates_after_mvp_ave[1000]))
    create_valence_plot_fig4_nolateral(x1=(rates_before_m4[1000],rates_before_mvp[1000]),
                                  x2=(rates_after_m4[1000],rates_after_mvp_app[1000]))

    print 'file saved'

    print 'M4 spiked '+ str( m4_spikes)
    print 'MVP spiked ' + str( mvp_spikes)


    return rates_before_m4, rates_after_m4, rates_after_m4_aversive, rates_before_mvp, \
           rates_after_mvp_app, rates_after_mvp_ave

def fig5_currents_before_after_aversive(w_mvp=[] , w_m4 = [] ,stdp_on =1):
    global w_kc_m4, mvp_gex, mvp_gotal, mvp_vinf, mvp_v, m4_gex, m4_ginh, m4_gtotal, m4_v, x_mr, y_mr, m4_spikes, \
        mtr_da_v,mtr_m4_t , w_kc_mvp, m4_dgex, SAVE_PLOTS, RUN_SIM, mtr_da_gex, \
        mtr_da_ginh
    global mvp_spikes, mvp_dgex
    global rates_before_m4,rates_after_m4,rates_after_m4_aversive , rates_before_mvp ,rates_after_mvp_app,\
        rates_after_mvp_ave
    #display.clear_output(wait=True)
    #time.sleep(1.0)
    d_ginh = 0.5
    m4_dgex = 0.5
    mvp_dgex = 0.5

    syns = [4,5]
    SAVE_PLOTS = 1
    restart_t =  0
    sp = 0
    #sp = np.zeros((duration ,))
    mvp_spikes = 0
    m4_spikes = 0
    if len(w_mvp) >0:
        w_kc_mvp = w_mvp
    if len(w_m4) > 0:
        w_kc_m4 = w_m4
    else:
        w_kc_m4 = np.random.random((n_kc ,))
    color_noise = colored_noise(50, dt, duration)
    mtr_m4_v[:] = m4_vrest
    mtr_m4_sp[:]  = 0
    mtr_gex[:]  = 0
    m4_noise = colored_noise(50, dt, duration)
    m4_noise = scale_linear_bycolumn(m4_noise, -2, 5.3)
    # odor exposure
    rates_before_m4 = np.zeros_like(mtr_m4_sp)
    rates_after_m4 = np.zeros_like(mtr_m4_sp)
    rates_after_m4_aversive = np.zeros_like(mtr_m4_sp)

    rates_before_mvp = np.zeros_like(mtr_m4_sp)
    rates_after_mvp_app = np.zeros_like(mtr_m4_sp)
    rates_after_mvp_ave = np.zeros_like(mtr_m4_sp)
    n_trials = 1
    #plot figures params
    fontsize = 9
    fontsize_l = 12
    lw = 0.5
    xlabels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 ]
    xticks = [ 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 ]
    yticks =[-60,-40,-20,0]
    size = (4,2)
    fig = plt.figure()
    fig.set_size_inches(size)
    lw = 0.5

    #before learning measure spikes
    for trials in range(0,n_trials):
        kcs[:,:]  = 0
        sp_times = np.sort(np.random.randint(low=start, high=m_duration, size=(n_kc_active,2)))
        sp_times2 = np.sort(np.random.randint(low=start, high=start+100 / dt, size=(n_kc_active,2)))
        for i in range(0, n_kc_active):
            kcs[sp_times[i],indices[i]] = 1
            kcs[sp_times2[i],indices[i]] = 1
        mtr_m4_sp[:] = 0
        mtr_mvp_sp[:] = 0
        for t in range(0,10000):
            mtr_w_m4[t % m_duration] = w_kc_m4
            mtr_xmr[t % m_duration] = x_mr
            mtr_ymr[t % m_duration] = y_mr
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
        #plt.figure()
        #plt.plot(mtr_gex,linewidth = 2, marker='', color ='darkgreen',label='Excitatory' )
        #plt.plot(-1 * mtr_ginh,linewidth = 2,marker='', color ='darkred',label='Inhibitory' )
        #plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=10 )
        #plt.title('before aversive learning')
        plt.ylim([-25,25])
        if trials == n_trials: #trial is the last one
            print 'last trial'
        rates_before_m4 = rates_before_m4 + lpf_spikes(mtr_m4_sp, timeshift = 500)
        rates_before_mvp = rates_before_mvp + lpf_spikes(mtr_mvp_sp, timeshift = 500)
    print 'this was mvp spikes before learnign'
    rates_before_m4 = rates_before_m4 / n_trials / 10
    rates_before_mvp = rates_before_mvp / n_trials / 10
    #perform aversive learning without lateral inhibition
    for trials in range(0,n_trials):
        kcs[:,:]  = 0
        sp_times = np.sort(np.random.randint(low=start, high=m_duration, size=(n_kc_active,2)))
        sp_times2 = np.sort(np.random.randint(low=start, high=start+100 / dt, size=(n_kc_active,2)))
        for i in range(0, n_kc_active):
            kcs[sp_times[i],indices[i]] = 1
            kcs[sp_times2[i],indices[i]] = 1
        w_kc_m4 = np.random.random((n_kc ,))
        w_kc_mvp = np.random.random((n_kc ,))
        m4_gex = 0
        mvp_gex = 0
        m4_v = m4_vrest
        mvp_v = mvp_vrest
        mtr_m4_sp[:] = 0
        for t in range(0,10000):
            mtr_w_m4[t % m_duration] = w_kc_m4
            mtr_w_mvp[t % m_duration] = w_kc_mvp
            mtr_xmr[t % m_duration] = x_mr
            mtr_ymr[t % m_duration] = y_mr
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
            #stdp
            d_x = kc_spikes_t
            d_y = m4_v == mvp_vrest
            x_mr = x_mr + dt * (-x_mr / tau_p + d_x)
            y_mr = y_mr + dt * ( -y_mr / tau_min + d_y)
            w_kc_m4 = w_kc_m4 - dt * (a_min * y_mr * d_x + a_plus * x_mr * d_y)
            ind_small = np.where ( w_kc_m4[:] < 0.1)
            w_kc_m4[ind_small]  = 0.1
        # odor retrieval after appetitive learning
        mtr_m4_sp[:] = 0
        mtr_mvp_sp[:] = 0
        for t in range(0,10000):
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
        rates_after_m4 = rates_after_m4 + lpf_spikes(mtr_m4_sp, timeshift = 500)
        rates_after_mvp_app = rates_after_mvp_app + lpf_spikes(mtr_mvp_sp, timeshift = 500)
    rates_after_m4 = rates_after_m4 / n_trials / 10
    rates_after_mvp_app = rates_after_mvp_app / n_trials / 10

    xlabels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 ]
    xticks = [ 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 ]
    plt.figure()
    fig.set_size_inches(size)
    ax = plt.gca()
    plt.subplot(121)
    ax = plt.gca()
    plt.plot(mtr_gex,linewidth = 0.2, marker='', color ='darkgreen',label='Excitatory' )
    plt.plot(-1 * mtr_ginh,linewidth = 0.2,marker='', color ='darkred',label='Inhibitory'  )
    plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=10 )
    plt.title('Before Aversive Learning', fontname="Times New Roman", fontsize =fontsize_l)
    plt.ylim([-5,5])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.yticks([-4,-3,-2,-1,0,1,2,3,4], fontsize = fontsize, fontname="Times New Roman")
    plt.xticks(xticks,xlabels, fontsize = fontsize,  fontname="Times New Roman")

    plt.subplot(122)
    ax = plt.gca()
    plt.plot(mtr_gex,linewidth = 0.2, marker='', color ='darkgreen',label='Excitatory' )
    plt.plot(-0.4 * mtr_ginh,linewidth = 0.2,marker='', color ='darkred',label='Inhibitory' )
    plt.legend( loc='upper right', numpoints = 1,frameon=False, fontsize=10 )
    plt.title('After Aversive Learning', fontname="Times New Roman", fontsize =fontsize_l)
    plt.ylim([-5,5])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.yticks([-4,-3,-2,-1,0,1,2,3,4], fontsize = fontsize, fontname="Times New Roman")
    plt.xticks(xticks,xlabels, fontsize = fontsize, fontname="Times New Roman")
    #now do aversive learning
    m4_dginh = 0.2
    for trials in range(0,n_trials):
        w_kc_m4 = np.random.random((n_kc ,))
        w_kc_mvp = np.random.random((n_kc ,))
        m4_gex = 0
        mvp_gex = 0
        m4_v = m4_vrest
        mvp_v = mvp_vrest
        mtr_m4_sp[:] = 0
        for t in range(0,10000):
            mtr_w_mvp[t % m_duration] = w_kc_mvp
            mtr_xmr[t % m_duration] = x_mr
            mtr_ymr[t % m_duration] = y_mr
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]
            d_x = kc_spikes_t
            d_y = mvp_v == mvp_vrest
            x_mr = x_mr + dt * (-x_mr / tau_p + d_x)
            y_mr = y_mr + dt * ( -y_mr / tau_min + d_y)
            w_kc_mvp = w_kc_mvp - dt * (-a_min * y_mr * d_x - a_plus * x_mr * d_y)
            #w_kc_m4 = w_kc_m4 - dt * (a_min * y_mr * d_x + a_plus * x_mr * d_y)
            ind_small = np.where ( w_kc_mvp[:] < 0.1)
            w_kc_mvp[ind_small]  = 0.1
        # odor retrieval after aversive learning
        mtr_m4_sp[:] = 0
        mtr_mvp_sp[:] = 0
        for t in range(0,10000):
            mvp_vrest = -60
            calculate_mvp(t)
            mtr_mvp_v[t % m_duration] = mvp_v
            calculate_m4(t, d_ginh)
            calculate_da(t)
            mtr_m4_v[t % m_duration] = m4_v
            mtr_da_v[t % m_duration] = da_v
            kc_spikes_t = kcs[t % m_duration,:]

    plt.savefig('fig5_before_after_aversive.eps',bbox_inches='tight', transparent="True", pad_inches=0)

    print 'file saved'
    print 'M4 spiked '+ str( m4_spikes)
    print 'MVP spiked ' + str( mvp_spikes)

    return rates_before_m4, rates_after_m4, rates_after_m4_aversive, rates_before_mvp, \
           rates_after_mvp_app, rates_after_mvp_ave

def fig5_before_after_aversive_fromdata():
    print 'todo'