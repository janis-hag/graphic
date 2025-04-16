import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def fancy_plot(x,y,title,xlabel,ylabel,leg=False, xlim=0, ylim=0, xscale='std', yscale='std', width=1, line_color='b', style="standard",grid=False,filename_to_save="No_saving",text=False,fig=True):
    params = {'legend.fontsize': 10}
    plt.rcParams['figure.figsize'] = 12, 10
    plt.rcParams.update(params)
    if fig:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    if np.size(np.shape(y))>1:
        for i in range(np.shape(y)[0]):
            plt.plot(x,y[i,:],linewidth=width)
    else:
        plt.plot(x,y,line_color,linewidth=width)
    if leg:
        plt.legend(leg)
    if text:
        ax.text(0.85, 0.7,'model used: \n Baraffe et al. 2015',fontsize=15, ha='center', va='center', transform=ax.transAxes)
    if xlim!=[0,0]:
        plt.xlim(xlim)
    if ylim!=[0,0]:
        plt.ylim(ylim)
    if xscale!='std':
        plt.xscale(xscale)
    if yscale!='std':
        plt.yscale(yscale)
    if style=="presentation":
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        plt.xlabel(xlabel).set_color('white')
        plt.ylabel(ylabel).set_color('white')
        plt.title(title).set_color('white')
        if grid:
            plt.grid(color='w')
        if filename_to_save!="No_saving":
            plt.savefig(filename_to_save,transparent=True,dpi=300)
    elif style=="standard":
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if grid:
            plt.grid()
        if filename_to_save!="No_saving":
            plt.savefig(filename_to_save,dpi=300)
