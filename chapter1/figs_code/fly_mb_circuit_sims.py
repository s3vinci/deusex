__author__ = 'MacbookPro'

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

def sim_mvp_m4_retrieval(w_mvp=[] , w_m4 = [] , d_ginh = 1,folder='fig2_results',stdp_on =1,
                         weights_color='purple'):
    global w_kc_m4, mvp_gex, mvp_gotal, mvp_vinf, mvp_v, m4_gex, m4_ginh, m4_gtotal, m4_v, x_mr, y_mr, m4_spikes, \
        mtr_da_v,mtr_m4_t , w_kc_mvp, m4_dgex, SAVE_PLOTS, RUN_SIM, mtr_da_gex, \
        mtr_da_ginh
    global mvp_spikes
    c_params = []
    t = 0
    print t
    print 'another second passedF'
    #plt.clf()
    display.clear_output(wait=True)
    #display.display(plt.gcf())
    time.sleep(1.0)
    file = open("params_values.txt", "r")
    val1 = file.readline().split()
    d_ginh = float(val1[1])
    m4_dgex = float(file.readline().split()[1])
    c_params = file.readline().split()
    syns = file.readline().split(',')
    syns = syns[1].split()
    syns = [int(i) for i in syns]
    prtz = float(file.readline().split()[1])
    SAVE_PLOTS = float(file.readline().split()[1])
    restart_t =  int(file.readline().split()[1])
    target_folder =  (file.readline().split()[1])
    stdp_inp = file.readline().split(' ')
    xticks_v = file.readline().split(',')
    xticks_v= xticks_v[1].split()
    xticks_t = [int(i) for i in xticks_v]
    yticks_v = file.readline().split(',')
    yticks_v= yticks_v[1].split()
    yticks_v = [int(i) for i in yticks_v]
    [stdp_on, restart_stdp] = int(stdp_inp[1]) , int(stdp_inp[3])
    file.close()
    print RUN_SIM
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
    #mtr_ginh[:]  = 0
    m4_noise = colored_noise(50, dt, duration)
    #v2_noise = colored_noise(50, Network.dt, Network.duration)
    m4_noise = scale_linear_bycolumn(m4_noise, -2, 5.3)
    #v2_noise = scale_linear_bycolumn(v2_noise, -2, 5.3)
    color_noise = scale_linear_bycolumn(color_noise, -2, 5.3)
    #kcs = np.zeros((m_duration,n_kc))
    #indices = np.sort(np.random.choice(n_kc, n_kc_active, replace=False))
    t = 0
    while RUN_SIM:
        if t < 5000:
            #kcs[t,:]  = 0
            m4_noise[t]  = 0
        #if t < 300:
        #m4_dgex = 0.01
        #fm4_noise[t] = 0
        #color_noise[t] = 0
        mtr_w_m4[t % m_duration] = w_kc_m4
        mtr_xmr[t % m_duration] = x_mr
        mtr_ymr[t % m_duration] = y_mr
        #color_noise[t] =  0
        #m4_noise[t]  =0
        mvp_vrest = -60
        #m4_noise[t]  =0
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
            print 'another second passedF32'
            #plt.clf()
            display.clear_output(wait=True)
            #display.display(plt.gcf())d3
            time.sleep(1.0)
            file = open("params_values.txt", "r")
            val1 = file.readline().split()
            d_ginh = float(val1[1])
            m4_dgex = float(file.readline().split()[1])
            c_params = file.readline().split()
            syns = file.readline().split(',')
            syns = syns[1].split()
            syns = [int(i) for i in syns]
            RUN_SIM = float(file.readline().split()[1])
            SAVE_PLOTS = float(file.readline().split()[1])
            restart_t =  int(file.readline().split()[1])
            target_folder =  (file.readline().split()[1])
            stdp_inp = file.readline().split(' ')
            xticks_v = file.readline().split(',')
            xticks_v= xticks_v[1].split()
            xticks_t = [int(i) for i in xticks_v]
            yticks_v = file.readline().split(',')
            yticks_v= yticks_v[1].split()
            yticks_v = [int(i) for i in yticks_v]
            [stdp_on, restart_stdp] = int(stdp_inp[1]) , int(stdp_inp[3])
            file.close()
            if restart_stdp:
                w_kc_m4 = np.random.random((n_kc ,))
            print 'but i made all the changwess'
            plot_it = 0
            if plot_it:
                #print 'ok'23
                raster_kc(spikes,size = (float ( c_params[5]), float( c_params[6])) ,lw=2,\
                                         fontsize=24,filename=folder+'/app_rasterkc')
                display.display(plt.gcf())
                plot_nice(mtr_m4_v, title = ' M4 V After Learning', color ='purple',
                          ylabel='V (mV)' , size = (float ( c_params[5]), float( c_params[6])),
                          xticks = xticks_t,yticks = yticks_v,filename=folder+'/app_v'+str(t),fontsize=24)
                display.display(plt.gcf())
                plot_currents(mtr_gex, mtr_ginh, title=' M4 Currents during learning',
                              color='darkblue', ylabel='nA',filename=folder+'/app_curr'+str(t),
                              ylims = (float(c_params[2]), float(c_params[3])),size=( float (c_params[5] ),
                                                                                      float(c_params[6])),
                              lw=2,fontsize=24,xticks=xticks_t )
                display.display(plt.gcf())
                plot_nice(mtr_w_m4[:, syns], title = 'Evolution of W',  size=(float(
                    c_params[5]), float(c_params[6])),fontsize=14,ylabel='w',xticks=xticks_t, yticks = [0,
                                                                                            0.5,
                                                                                            0.8],
                                                                                            color=weights_color,
                          filename=folder+'/app_w'+str(t))
                display.display(plt.gcf())
                plot_nice(mtr_ymr[:, syns],fontsize=24, title = 'Postsynaptic  eligibility trace' ,
                          size = (float ( c_params[5]), float( c_params[6])),
                          xticks=xticks_t,ylabel='DA1',filename=folder+'/DA1')
                display.display(plt.gcf())
                plot_nice(mtr_xmr[:, syns],fontsize=24, title = 'Presynaptic eligibility trace', size=(float(
                    c_params[5]), float(c_params[6])),xticks=xticks_t,
                          ylabel='pre-post',filename=folder+'/prepost')
                display.display(plt.gcf())
                plot_nice(mtr_da_v, title = ' DA V', color ='green',
                          ylabel='V (mV)' , fontsize=24,size = (float ( c_params[5]), float( c_params[6])),
                          xticks=xticks_t,
                          yticks=yticks_v,filename=folder+'/DAv')
                display.display(plt.gcf())
                plot_currents(mtr_da_gex, mtr_da_ginh, title=' DA Currents during learning',
                              color='darkblue', ylabel='nA',
                              filename=folder+'/app_currDAv'+str(t),
                              ylims = (float(c_params[2]), float(c_params[3])),size=( float (c_params[5] ), float(c_params[6])),lw=2,fontsize=14 )
                display.display(plt.gcf())
            file.close()
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


def sim_mvp_m4_appetitive_learn(w_stdp=[] , stdp  = 0, d_ginh = 1):
    global w_kc_m4, mvp_gex, mvp_gotal, mvp_vinf, mvp_v, m4_gex, m4_ginh, m4_gtotal, m4_v, x_mr, y_mr, m4_spikes, \
        mtr_m4_t , w_kc_mvp
    global mvp_spikes, color_noise, m4_noise
    #v2_noise = scale_linear_bycolumn(v2_noise, -2, 5.3)
    if len(w_stdp) !=0:
        w_kc_m4 = w_stdp
        print 'gave values'
    plt.figure(1)
    raster_kc(spikes)
    #print mtr_m4_tk
    kcs = np.zeros((duration,n_kc))
    #kc_spikes_t = np.zeros((n_kc,))
    indices = np.sort(np.random.choice(n_kc, n_kc_active, replace=False))
    sp_times = np.sort(np.random.randint(low=start, high=end, size=(n_kc_active,2)))
    sp_times2 = np.sort(np.random.randint(low=start, high=start+100 / dt, size=(n_kc_active,2)))
    for i in range(0, n_kc_active):
        kcs[sp_times[i],indices[i]] = 1
        kcs[sp_times2[i],indices[i]] = 1
    for t in range (0 ,int(duration)):
        kc_spikes_t = kcs[t,:]
        mtr_w_m4[t] = w_kc_m4
        mtr_xmr[t] = x_mr
        mtr_ymr[t] = y_mr
        color_noise[t] =  0
        m4_noise[t] =0
        calculate_mvp(t)
        mtr_mvp_v[t]  = mvp_v
        calculate_m4(t, d_ginh)
        mtr_m4_v[t]  = m4_v
        d_x = kc_spikes_t
        d_y = mvp_v == mvp_vrest
        x_mr = x_mr + dt * (-x_mr / tau_p + d_x)
        y_mr = y_mr + dt * ( -y_mr / tau_min + d_y)
        if stdp:
            w_kc_m4 = w_kc_m4 - dt * (a_min * y_mr * d_x + a_plus * x_mr * d_y)
    print 'M4 spiked '+ str ( m4_spikes)
    print 'MVP spiked ' + str( mvp_spikes)
    return mtr_m4_sp, w_kc_m4
def sim_mvp_m4_aversive_learn(w_stdp=[] , stdp  = 0, d_ginh = 1, show_plots = 1):
    global w_kc_m4, mvp_gex, mvp_gotal, mvp_vinf, mvp_v, m4_gex, m4_ginh, m4_gtotal, m4_v, x_mr, y_mr, m4_spikes, \
        mtr_m4_t, w_kc_mvp, kcs
    global mvp_spikes
    #color_noise = colored_noise(50, dt, duration)
    #m4_noise = colored_noise(50, dt, duration)
    #v2_noise = colored_noise(50, Network.dt, Network.duration) 2
    #m4_noise = scale_linear_bycolumn(m4_noise, -2, 5.3)
    #v2_noise = scale_linear_bycolumn(v2_noise, -2, 5.3)
    #color_noise = scale_linear_bycolumn(color_noise, -2, 5.3)
    if len(w_stdp) !=0:
        w_kc_m4 = w_stdp
        print 'gave values'
    plt.figure(1)
    raster_kc(spikes)
    #kcs = np.zeros((duration,n_kc))
    indices = np.sort(np.random.choice(n_kc, n_kc_active, replace=False))
    sp_times = np.sort(np.random.randint(low=start, high=end, size=(n_kc_active,2)))
    sp_times2 = np.sort(np.random.randint(low=start, high=start+100 / dt, size=(n_kc_active,2)))
    for i in range(0, n_kc_active):
        kcs[sp_times[i],indices[i]] = 1
        kcs[sp_times2[i],indices[i]] = 1
    print 'starting simulation'
    for t in range (0 , int(duration)):
        if t % 1000 == 0:
            print t
        mtr_w_m4[t] = w_kc_mvp
        color_noise[t] =  0
        m4_noise[t]  =0
        calculate_mvp(t)
        mtr_mvp_v[t]  = mvp_v
        calculate_m4(t, d_ginh)
        d_x = kc_spikes_t
        d_y = mvp_v == mvp_vrest
        x_mr = x_mr + dt * (-x_mr / tau_p + d_x)
        y_mr = y_mr + dt * ( -y_mr / tau_min + d_y)
        if stdp:
            w_kc_mvp = w_kc_mvp - dt * (a_min * y_mr * d_x + a_plus * x_mr * d_y)
    #plt.figure(3)
    print 'M4 spiked '+ str ( m4_spikes)
    print 'MVP spiked ' + str( mvp_spikes)
    #if show_plots:
    #plot_neuron_spikes()
    return mtr_m4_sp, w_kc_mvp

def check_appetitive_learn():
        [sp, w_neutral] = sim_mvp_m4_retrieval(d_ginh=0.1)
        rates_before = lpf_spikes(mtr_m4_sp)
        plot_currents(mtr_gex, mtr_ginh, title='Currents before learning',color='blue'  )
        plot_nice(w_neutral, title = 'M4 V Before Learning', color ='blue')
        plot_nice(w_kc_m4  , title = 'KC - M4 Weights before learning' , color='blue')

        [sp, w_app] = sim_mvp_m4_appetitive_learn(d_ginh=0.1, stdp = 1)
        plot_nice(mtr_m4_v, title = 'M4 V After Learning', color ='blue')
        plot_nice(w_app ,title = 'KC - M4 Weights after learning', color = 'blue')
        plot_currents(mtr_gex, mtr_ginh, title='Currents after learning'  )
        [sp, w] = sim_mvp_m4_retrieval(w_m4 = w_app , d_ginh=0.1)
        rates_after = lpf_spikes(mtr_m4_sp)
        plot_avg_before_after(rates_before,rates_after, title= ' Firing rate before ('
                                                               ' green) / after '
                                                               'appetitive '
                                                               'learning (red')