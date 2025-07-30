import numpy as np
import matplotlib.pyplot as plt

def func(x): return x ** 2 #np.square(x)

# the first-order derivative : dy/dx=2*x
# 函數x ** 2 微分, 變成 2 * x
def dfunc(x): return 2 * x

# Gradient Descent
def GD(x_start, df, epochs, lr):
    '''
    Initialization of Variables
    '''
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    '''
    Gradient Descent Loop
    '''
    for i in range(epochs):
        dx = df(x)
        # 的微分 * learning rate
        x -= dx * lr
        xs[i+1] = x
#     retun 2d metrix.
    return xs

x_start = 5     # initial weight
epochs = 15
lr = 0.3        # learning rate
w = GD(x_start, dfunc, epochs, lr=lr)
print(w)
'''
Plotting the Loss Function and Tangents
'''
plt.figure(figsize=(12,8))
# 尺度: -6 ~ 6
t = np.arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')

plt.title('Gradient Descent', fontsize=20)
plt.xlabel('w', fontsize=20)
plt.ylabel('Loss', fontsize=20)

color = list('rgbymrgbymrgbym') # many colorful lines
line_offset=2           #length of the tangent line

# Draw a lot of line to get closer to the very bottom of the loss.
# Every epoch creates a tangent line
for i in range(epochs):
    # ex: Array[5, 5.0001]
    z=np.array([w[i], w[i]+0.001])
    print("z=", z)
    vec=np.vectorize(func)
    print("vec(z)=", vec(z))
    p = np.polyfit(z, vec(z), deg=1)
    print("p=", p)
    '''
    Generate X and Y Coordinates for the Tangent Line
    '''
    x=np.array([w[i]-line_offset, w[i]+line_offset])
    # p[0]: 一次向係數, p[1]: 常數向係數
    y=np.array([(w[i]-line_offset)*p[0]+p[1], (w[i]+line_offset)*p[0]+p[1]])
    # draw line: x axis, y axis, color of  line
    plt.plot(x, y, c=color[i-1]) #i-1: 從最後一個顏色開始 (m)