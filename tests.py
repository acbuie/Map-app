# import time
# from matplotlib import pyplot as plt
# import numpy as np


# def live_update_demo(blit = False):
#     x = np.linspace(0,50., num=100)
#     X,Y = np.meshgrid(x,x)
#     fig = plt.figure()
#     ax1 = fig.add_subplot(2, 1, 1)
#     ax2 = fig.add_subplot(2, 1, 2)

#     img = ax1.imshow(X, vmin=-1, vmax=1, interpolation="None", cmap="RdBu")


#     line, = ax2.plot([], lw=3)
#     text = ax2.text(0.8,0.5, "")

#     ax2.set_xlim(x.min(), x.max())
#     ax2.set_ylim([-1.1, 1.1])

#     fig.canvas.draw()   # note that the first draw comes before setting data 


#     if blit:
#         # cache the background
#         axbackground = fig.canvas.copy_from_bbox(ax1.bbox)
#         ax2background = fig.canvas.copy_from_bbox(ax2.bbox)

#     plt.show(block=False)


#     t_start = time.time()
#     k=0.

#     for i in np.arange(1000):
#         img.set_data(np.sin(X/3.+k)*np.cos(Y/3.+k))
#         line.set_data(x, np.sin(x/3.+k))
#         tx = 'Mean Frame Rate:\n {fps:.3f}FPS'.format(fps= ((i+1) / (time.time() - t_start)) ) 
#         text.set_text(tx)
#         #print tx
#         k+=0.11
#         if blit:
#             # restore background
#             fig.canvas.restore_region(axbackground)
#             fig.canvas.restore_region(ax2background)

#             # redraw just the points
#             ax1.draw_artist(img)
#             ax2.draw_artist(line)
#             ax2.draw_artist(text)

#             # fill in the axes rectangle
#             fig.canvas.blit(ax1.bbox)
#             fig.canvas.blit(ax2.bbox)

#             # in this post http://bastibe.de/2013-05-30-speeding-up-matplotlib.html
#             # it is mentionned that blit causes strong memory leakage. 
#             # however, I did not observe that.

#         else:
#             # redraw everything
#             fig.canvas.draw()

#         fig.canvas.flush_events()
#         #alternatively you could use
#         #plt.pause(0.000000000001) 
#         # however plt.pause calls canvas.draw(), as can be read here:
#         #http://bastibe.de/2013-05-30-speeding-up-matplotlib.html


# live_update_demo(True)   # 175 fps
# #live_update_demo(False) # 28 fps

import numpy as np
import time
import matplotlib
matplotlib.use('GTKAgg')
from matplotlib import pyplot as plt


def randomwalk(dims=(256, 256), n=20, sigma=5, alpha=0.95, seed=1):
    """ A simple random walk with memory """

    r, c = dims
    gen = np.random.RandomState(seed)
    pos = gen.rand(2, n) * ((r,), (c,))
    old_delta = gen.randn(2, n) * sigma

    while True:
        delta = (1. - alpha) * gen.randn(2, n) * sigma + alpha * old_delta
        pos += delta
        for ii in xrange(n):
            if not (0. <= pos[0, ii] < r):
                pos[0, ii] = abs(pos[0, ii] % r)
            if not (0. <= pos[1, ii] < c):
                pos[1, ii] = abs(pos[1, ii] % c)
        old_delta = delta
        yield pos


def run(niter=1000, doblit=True):
    """
    Display the simulation using matplotlib, optionally using blit for speed
    """

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.hold(True)
    rw = randomwalk()
    x, y = rw.next()

    plt.show(False)
    plt.draw()

    if doblit:
        # cache the background
        background = fig.canvas.copy_from_bbox(ax.bbox)

    points = ax.plot(x, y, 'o')[0]
    tic = time.time()

    for ii in xrange(niter):

        # update the xy data
        x, y = rw.next()
        points.set_data(x, y)

        if doblit:
            # restore background
            fig.canvas.restore_region(background)

            # redraw just the points
            ax.draw_artist(points)

            # fill in the axes rectangle
            fig.canvas.blit(ax.bbox)

        else:
            # redraw everything
            fig.canvas.draw()

    plt.close(fig)
    print "Blit = %s, average FPS: %.2f" % (
        str(doblit), niter / (time.time() - tic))

if __name__ == '__main__':
    run(doblit=False)
    run(doblit=True)