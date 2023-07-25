import numpy as np
from initialization import xf0, yf0, xb0, yb0

x = 0.254
if xb0[0] < x < xb0[-1]:
        i = int(x/0.005)
        print(i)
        dx = xb0[i] - xb0[i-1]
        dy = yb0[i] - yb0[i-1]
        slope = dy / dx
        x_interp = yb0[i-1] + slope * (x - xb0[i-1])
        print(xb0[i-1], x, xb0[i])
        print(yb0[i-1], x_interp, yb0[i])

    #elif x < xbmin:
       # return Jb_low(x)
    #else:
        #return Jb_high(x)
