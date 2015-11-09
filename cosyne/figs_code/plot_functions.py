import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter


# other helper functions
def raster_plot_save(all_kc, filename='kc_raster'):
    ax = plt.gca()
    color = 'k'
    for ith, trial in enumerate(all_kc):
        plt.vlines(trial, ith + .9, ith + 15.5, color=color)
    plt.ylim(2.5, len(all_kc) + 2.5)
    plt.xlim(0, 20000)
    plt.ylabel('KC index', fontsize=13)
    plt.xlabel('Time (s)', fontsize=13)
    # lt.ylabel('Membrance Potential (mV)',fontsize=13)
    # plt.xlabel('Time (s)',fontsize=13)
    plt.tick_params(labelsize=13)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.savefig('results/' + filename)
def raster(event_times_list, color='b'):
    """
    Creates a raster plot
    Parameters
    ----------
    event_times_list : iterable
                       a list of event time iterables
    color : string
            color of vlines
    Returns
    -------
    ax : an axis containing the raster plot
    """
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 1.5, color=color)
    plt.ylim(.5, len(event_times_list) + .5)
    return ax
def calc_fire_rate2(trials, duration=10000, timeshift=10):
    rates = []
    for sp in trials:
        scaling_factor = duration / timeshift
        temp = []
        for window in range(0,len(sp)):
            n_spikes  = np.sum( sp[0+window:timeshift+window])
            rate = n_spikes * scaling_factor
            temp.append(rate)
        rates.append(temp)
    return rates
def lpf_spikes(spikes, duration=10000, timeshift=3200):
    scaling_factor = duration / timeshift
    r_hz = []
    for window in range(0,len(spikes)):
        n_spikes  = np.sum( spikes[0+window:timeshift+window])
        rate = n_spikes * scaling_factor
        r_hz.append(rate)
    return r_hz
def get_avg_firing_rate_over_many_trials(w_kc_m4, w_kc_mvp, iters = 4 , appetitive_learn = 1, aversive_learn = 0):
    if appetitive_learn:
        print 'app'
    if aversive_learn:
        print 'avv'

def plot_stdp_exp():
    size = [6,4]
    fontsize = 14
    time = [-25,-20,-14,-10,0,10,15,20]
    stdp_points = [0,-1,-5, -1,-4, 60,5,1 ]
    oct_points =  [0,-5,-30,-1,-4,-8,-10,1]
    fig = plt.figure()
    ax = plt.gca()
    fig.set_size_inches(size)
    plt.plot(time,stdp_points,linewidth = 2, color = 'grey' ,marker="o")
    plt.plot(time,oct_points,linewidth = 2, color = 'blue',marker="o")
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.gcf().subplots_adjust(bottom=.4,left=.2)
    plt.ylabel('Change %', fontsize=13)
    plt.xlabel('Time (s)', fontsize=13)
    plt.yticks([-50,0,50])
    plt.xticks([-20,-15,-10,-5,0,5,10,15,20])
    plt.ylim([-60,60])
def plot_stdp_model():
    fig = plt.figure()
    ax = plt.gca()
    fig.set_size_inches(size)
    if dims == 1:
        plt.plot(data,linewidth = lw, color =color ,marker=marker)
    else:
        plt.plot(data[0],data[1], linewidth = lw, color=color)
    if not ARTISTIC_PLOT:
        plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    if SHOW_TITLE:
        plt.title(title,fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.gcf().subplots_adjust(bottom=.4,left=.2)
def plot_all():
    t = 0
    xticks_t = [0 , 2000, 4000 , 6000 , 8000 ,10000]
    yticks_v = [-60 , -40 , -20, 0]
    size = [10,5]
    syns = [6,10,15]
    weights_color = 'purple'
    folder = 'new_results/'
    raster_kc(spikes,size = (float ( size[0]), float(size[1])) ,lw=2,\
                             fontsize=24,filename=folder+'/app_rasterkc')
    display.display(plt.gcf())
    plot_nice(mtr_m4_v, title = ' M4 V After Learning', color ='purple',
              ylabel='V (mV)' , size = (float (size[0]), float( size[1])),
              xticks = xticks_t,yticks = yticks_v,filename=folder+'/app_v'+str(t),fontsize=24)
    display.display(plt.gcf())
    plot_currents(mtr_gex, mtr_ginh, title=' M4 Currents during learning',
                  color='darkblue', ylabel='nA',filename=folder+'/app_curr'+str(t),
                  ylims = (-1,5),size=( float (size[0] ),
                                                                          float(size[1])),
                  lw=2,fontsize=24,xticks=xticks_t )
    display.display(plt.gcf())
    plot_nice(mtr_w_m4[:, syns], title = 'Evolution of W',  size=(float(
        size[0]), float(size[1])),fontsize=14,ylabel='w',xticks=xticks_t, yticks = [0,
                                                                                0.5,
                                                                                0.8],
                                                                                color=weights_color,
              filename=folder+'/app_w'+str(t))
    display.display(plt.gcf())
    plot_nice(mtr_ymr[:, syns],fontsize=24, title = 'Postsynaptic  eligibility trace' ,
              size = (float ( size[0]), float( size[1])),
              xticks=xticks_t,ylabel='DA1',filename=folder+'/DA1')
    display.display(plt.gcf())
    plot_nice(mtr_xmr[:, syns],fontsize=24, title = 'Presynaptic eligibility trace', size=(float(
        size[0]), float(size[1])),xticks=xticks_t,
              ylabel='pre-post',filename=folder+'/prepost')
    display.display(plt.gcf())
    plot_nice(mtr_da_v, title = ' DA V', color ='green',
              ylabel='V (mV)' , fontsize=24,size = (float(size[0]), float( size[1])),
              xticks=xticks_t,
              yticks=yticks_v,filename=folder+'/DAv')
    display.display(plt.gcf())
    plot_currents(mtr_da_gex, mtr_da_ginh, title=' DA Currents during learning',
                  color='darkblue', ylabel='nA',
                  filename=folder+'/app_currDAv'+str(t),
                  ylims = (-1,5),size=( float (size[0]), float(size[1])),
                  lw=2,fontsize=14 )
    display.display(plt.gcf())

def scale_linear_bycolumn(rawpoints, high=10.0, low=-10.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)
def white_noise(dt, duration, dims=1):
    n = int(duration / dt)
    noise = np.random.normal(0, 1, (n, 1))
    return noise
def colored_noise(tau, dt, duration, dims=1):
    noise = white_noise(dt, duration, dims)
    a = [1., -np.exp(-dt / tau)]
    b = [1.]
    fnoise = np.sqrt(2 * dt / tau) * lfilter(b, a, noise)
    return fnoise

def plot_currents(gex,ginh , xlabel = 'Time (s)',ylabel='ylabel',title='Title',color='r',
                  filename='currents',
                  size=(10, 5),fontsize= 10,lw=2,marker='',ylims=(-5,5),xticks=[],yticks=[]):
    fig = plt.figure()
    ax = plt.gca()
    fig.set_size_inches(size)
    plt.plot(gex,linewidth = lw, marker=marker, color ='darkgreen' )
    plt.plot(-1 * ginh,linewidth = lw,marker=marker, color ='darkred' )
    if not ARTISTIC_PLOT:
        plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    if SHOW_TITLE:
        plt.title(title, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.gcf().subplots_adjust(bottom=.4,left=.2)
    plt.xlim([0,10000])
    plt.ylim(ylims)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if len (xticks ) > 0:
        plt.xticks(xticks,xlabels)
    if len (yticks ) > 0:
        plt.yticks(yticks)
    if ARTISTIC_PLOT:
        ax.spines['bottom'].set_color('none')
        plt.xticks([])
    if SAVE_PLOTS:
        plt.savefig('results/' + filename+'.eps')
        plt.savefig('results/' + filename+'.png')
def plot_neuron_spikes():
    fig = plt.figure(3)
    tmp = []
    tmp.append(mtr_m4_t)
    plt.figure()
    raster(tmp)
    plt.title('M4 spikes' ,fontsize=30)
    fig.set_size_inches(20,10)
    plt.show()
    tmp = []
    tmp.append(mtr_mvp_t)
    fig = plt.figure(15)
    raster(tmp)
    plt.title('MVP spikes')
    fig.set_size_inches(20,10)
    plt.show()
    print mvp_spikes
    #plt.figure(11)
    #for i in range(0,100):
    #    plt.plot(mtr_w_m4[:,i])
    #plt.show()

#helper functions

def plot_weights_before_after(data1, data2,  xlabel =  '#KC - M4 synapse',ylabel='W',
                              title='Title',color1='black',
                              color2='r',filename='weightsbeforeafter'):
    fig = plt.figure()
    ax = plt.gca()
    fig.set_size_inches(20,10)
    #print len(data)
    plt.scatter(np.linspace(1,100, 100),  data1,linewidth = 4,marker='o',  color =color1 )
    plt.scatter(np.linspace(1,100, 100),  data2,linewidth = 4,marker='o',  color =color2 )
    plt.xlabel(xlabel,fontsize=30)
    plt.ylabel(ylabel,fontsize=30)
    plt.xlim([0,100])
    #plt.title(title, fontsize = 30)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.tick_params(labelsize=30)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    #plt.xlim([0,10000]) 4224
    if SAVE_PLOTS:
        plt.savefig('results/' + filename+'.eps')
        plt.savefig('results/' + filename+'.png')
def plot_avg_before_after(gex,ginh , xlabel = ' Time (s)',ylabel='ylabel',
                          title='Title',color='r',
                          filename=' hzbeforeafter', size=(10, 5),fontsize= 10,lw=2,
                          marker=''):
    fig = plt.figure()
    ax = plt.gca()
    fig.set_size_inches(size)
    #print len(data)
    plt.plot(gex,linewidth = 4,marker=marker, color ='black' )
    plt.plot( ginh,linewidth = 4,marker='o', color =color )
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.title(title,fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlim([0,10000])
    plt.gcf().subplots_adjust(bottom=.4,left=.2)
    plt.savefig('results/' + filename)
    plt.savefig('results/' + filename+'.eps')
    plt.savefig('results/' + filename+'.png')
def plot_nice(data, xlabel = 'Time (s)',ylabel='ylabel',title='Title',color='b', dims = 1, filename='nice',
              size=(10,5), fontsize= 10,lw=2,marker='',xticks=[],yticks=[]):
    fig = plt.figure()
    ax = plt.gca()
    fig.set_size_inches(size)
    if dims == 1:
        plt.plot(data,linewidth = lw, color =color ,marker=marker)
    else:
        plt.plot(data[0],data[1], linewidth = lw, color=color)
    if not ARTISTIC_PLOT:
        plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    if SHOW_TITLE:
        plt.title(title,fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.gcf().subplots_adjust(bottom=.4,left=.2)

    if ARTISTIC_PLOT:
        ax.spines['bottom'].set_color('none')
    if len (xticks ) > 0 and not ARTISTIC_PLOT:
        plt.xticks(xticks,xlabels)
    #if len (yticks ) > 0:
    plt.yticks(yticks)
    if ARTISTIC_PLOT:
        ax.spines['bottom'].set_color('none')
        plt.xticks([])
    if SAVE_PLOTS:
        plt.savefig('results/' + filename+'.eps')
        plt.savefig('results/' + filename+'.png')
def raster_kc(event_times_list, color='k', filename = 'kc_raster',ilename='currents',
                  size=(10, 5),fontsize= 10,lw=2,marker=''):
    """
    Creates a raster plot
    Parameters
    ----------
    event_times_list : iterable
                       a list of event time iterables
    color : string
            color of vlines
    Returns
    -------
    ax : an axis containing the raster plot
    """
    fig = plt.figure()
    ax = plt.gca()
    fig.set_size_inches(size)
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial[0], ith + .5, ith + 1.5, color=color, linewidth = lw)
    plt.ylim(.5, len(event_times_list) + .5)
    plt.tick_params(labelsize=fontsize)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.xlim([0,10000])
    plt.ylabel('KC #No', fontsize=fontsize)
    plt.xlabel('Time (s)',fontsize=fontsize)
    plt.gcf().subplots_adjust(bottom=.4,left=.2)
    if ARTISTIC_PLOT:
        ax.spines['bottom'].set_color('none')
        plt.xticks([])
        plt.xlabel('',fontsize=fontsize)
    plt.savefig('results/' + filename+'.eps')
    plt.savefig('results/' + filename+'.png')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return ax




def scale_linear_bycolumn(rawpoints, high=10.0, low=-10.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)
def white_noise(dt, duration, dims=1):
    n = int(duration / dt)
    noise = np.random.normal(0, 1, (n, 1))
    return noise
def colored_noise(tau, dt, duration, dims=1):
    noise = white_noise(dt, duration, dims)
    a = [1., -np.exp(-dt / tau)]
    b = [1.]
    fnoise = np.sqrt(2 * dt / tau) * lfilter(b, a, noise)
    return fnoise
