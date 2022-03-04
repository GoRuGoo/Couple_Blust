import matplotlib as plt
import time
import numpy as np
nowtime = int(time.time())

def plot_graph(y):
    fig,ax = plt.subplots(1,1)
    ax.set_ylim((0,1000))
    x = np.arrange(0,100,0.5)
    line, = ax.plot(x,y,color='blue')
    plt.pause(0.01)
    line.remove()
