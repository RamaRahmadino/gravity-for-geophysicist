import numpy as np
import matplotlib.pyplot as plt

def HorizontalGradeintMagnitude(x,y,CBA):
    karnelx = np.array([[0, 0, 0],
                        [-0.5, 0, 0.5],
                        [0, 0, 0]])
    from scipy.signal import convolve2d
    resultx = convolve2d(CBA, karnelx, mode='valid', boundary='symm')
    # print("resulty", resultx.shape)
    resultx = resultx ** 2
    # # Koordinat arah X
    index = [0, -1]
    x_first = np.delete(x, index)

    karnely = np.array([[0,-0.5,0],
                         [0,0,0],
                         [0,0.5,0]])

    from scipy.signal import convolve2d
    resulty = convolve2d(CBA, karnely, mode='valid', boundary='symm')
    resulty = resulty ** 2
    index = [0, -1]
    y_first = np.delete(y, index)

    #Final Script
    g_first = np.sqrt(resultx + resulty)
    fig, ax_1 = plt.subplots(figsize=(12, 10))
    im = ax_1.contourf(x_first, y_first, g_first, levels=40, cmap = "turbo")
    plt.title('Horizontal Gradient Magnitude')
    fig.colorbar(im, label = "mGal")
    plt.xlabel("Easthing (m)")
    plt.ylabel("Northing (m)")
    plt.show()
    return x_first, y_first, g_first

def SecondVerticalDerivative(x,y,CBA):
    def x_direction(x,y,CBA):
        karnelx = np.array([[0, -0.5, 0],
                            [0, 0, 0],
                            [0, 0.5, 0]])
        from scipy.signal import convolve2d
        resultx = convolve2d(CBA, karnelx, mode='valid', boundary='symm')
        # print("resultx", resultx.shape)
        fdx2 = convolve2d(resultx, karnelx, mode='valid', boundary='symm')
        index = [0, 1, -2, -1]
        x_first2 = np.delete(x, index)
        return x_first2, fdx2

    x_first2, fdx2 = x_direction(x,y,CBA)
    def y_direction (x, y, CBA):
        karnely = np.array([[0, 0, 0],
                            [-0.5, 0, 0.5],
                            [0, 0, 0]])
        from scipy.signal import convolve2d
        resulty = convolve2d(CBA, karnely, mode='valid', boundary='symm')
        # print("resulty", resulty.shape)
        fdy2 = convolve2d(resulty, karnely, mode='valid', boundary='symm')
        # print("resluty_2",  fdy2.shape)
        # Koordinat arah y
        index = [0, 1, -2, -1]
        y_first2 = np.delete(y, index)
        return y_first2, fdy2

    y_first2, fdy2 = y_direction(x, y, CBA)

    v_svd = -(fdx2 + fdy2)
    fig, ax_1 = plt.subplots(figsize=(12, 10))
    im = ax_1.contourf(x_first2, y_first2, v_svd, levels=40, cmap="turbo")
    plt.title('Second Vertival Derivative')
    fig.colorbar(im, label = "mGal")
    plt.xlabel("Easthing (m)")
    plt.ylabel("Northing (m)")
    plt.show()

    return  x_first2, y_first2, v_svd
