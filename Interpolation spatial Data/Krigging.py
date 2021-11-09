import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gstools as gs
from pykrige.ok import OrdinaryKriging
import pykrige.kriging_tools as kt

##Define data
data = np.loadtxt('C:/Users/hp/PycharmProjects/pythonProject/realdata_1.dat')
utm_x = np.array(data[:,0])
utm_y = np.array(data[:,1])
utm_z = np.array(data[:,2])
gobs = np.array(data[:,3])
CBA = np.array(data[:,4])

#Define Grid
space = 200
Nx = (max(utm_x)-min(utm_x))/space +1
Ny = (max(utm_y)-min(utm_y))/space +1
x = np.linspace(min(utm_x), max(utm_x), int(Nx))
y = np.linspace(min(utm_y), max(utm_y), int(Ny))

lag = np.linspace(0,20000,100) #data real
lag_center, gamma = gs.vario_estimate_unstructured([utm_x,utm_y],CBA,lag)

def model (model, leg, var, nugget):
    if model == 1:
        model_ = gs.Spherical(dim=2, len_scale=leg, var=var, nugget=nugget) #model spherical
    elif model == 2:
        model_ = gs.Exponential(dim=2, len_scale=leg, var=var, nugget=nugget) #model exponential
    elif model == 3:
        model_ = gs.Gaussian(dim=2, len_scale=leg, var=var, nugget=nugget) #model gaussian
    else:
        print("model tidak tersedia")

    model_.plot(x_max = 20000)
    plt.plot(lag_center,gamma, '.')
    plt.show()
    return model_

#Define model
model = model(1, 11000, 26, 0.1)

pk_kwargs = model.pykrige_kwargs
OK = OrdinaryKriging(utm_x, utm_y, CBA, **pk_kwargs)
zi, ss1 = OK.execute("grid", x, y)
xi, yi = np.meshgrid(x, y)
fig, ax_1 = plt.subplots(figsize=(12,10))
im_1 = ax_1.contourf(xi,yi,zi,levels=40, cmap="jet")

plt.xlabel("Easthing (m)")
plt.ylabel("Northing (m)")
ax_1.scatter(utm_x,utm_y, linewidths = 0.0005, label = "Stasiun gayaberat")
fig.colorbar(im_1, label = "mGal")
plt.title("Complete Bouguer Anomaly (CBA)")
plt.show()