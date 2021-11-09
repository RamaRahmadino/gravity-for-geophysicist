import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def SpectrumAnalysis1D (xi ,yi ,zi, loc, Slicing, space):
  answare1 = 'Y'
  while(True):
    answare2 = 'Y'
    while(True):
      if Slicing == 1:
        # N-S
        x_slice = xi[loc]
        # print('coordinate in X diractions :', x_slice)

        y_slice = yi
        # print('Coordinate in Y direction:', y_slice)

        g_slice = []
        for k in range(len(yi)):
          gslice = zi[k][loc]
          g_slice.append(float(gslice))
        g_slice = np.array(g_slice)

        data_new = pd.DataFrame({"UTM_X": x_slice, "UTM_Y": y_slice, "CBA": g_slice})
        fig, ax_1 = plt.subplots(figsize=(12, 10))
        im_1 = ax_1.contourf(xi, yi, zi, levels=40, cmap="jet")
        plt.xlabel("Easthing (m)")
        plt.ylabel("Northing (m)")
        fig.colorbar(im_1, label="mGal")
        plt.title("Complete Bouguer Anomaly (CBA)")
        im_1 = plt.plot(data_new.UTM_X, data_new.UTM_Y)
        plt.show()

      elif Slicing == 2:
        # E-W
        x_slice = xi
        # print('Coordinate in X direction:', x_slice)

        y_slice = yi[loc]
        # print('Coordinate slice in Y direction:', y_slice)

        g_slice = zi[loc]

        data_NS = pd.DataFrame({"UTM_X": x_slice, "UTM_Y": y_slice, "CBA": g_slice})

        fig, ax_1 = plt.subplots(figsize=(12, 10))
        im_1 = ax_1.contourf(xi, yi, zi, levels=40, cmap="turbo")
        plt.xlabel("Easthing (m)")
        plt.ylabel("Northing (m)")
        fig.colorbar(im_1, label="mGal")
        plt.title("Complete Bouguer Anomaly (CBA)")
        ax_1.plot(data_new.UTM_X, data_NS.UTM_Y)
        plt.show()

      else:
        print("Wrong slicing direction, check instructions")

      ### calculate interval data
      def interval(DF_Data):
        DF_Data.dropna(inplace=True)
        DF_Data = DF_Data.reset_index(drop=True)

        dist_x = np.array([j - i for i, j in zip(DF_Data.UTM_X[:-1], DF_Data.UTM_X[1:])])
        dist_y = np.array([j - i for i, j in zip(DF_Data.UTM_Y[:-1], DF_Data.UTM_Y[1:])])
        dist = np.sqrt((dist_x ** 2) + (dist_y ** 2))
        dist = np.cumsum(dist)
        dist = np.append(0, dist)
        return dist

      dist = interval(data_new)
      ### Ln A Absolute
      def fouier (CBA):
        V_CBA = np.fft.fft(CBA)
        imaginary = V_CBA.imag
        real = V_CBA.real
        absolute = np.sqrt((imaginary ** 2)+ (real ** 2))
        absolute = absolute[0:int(np.floor(len(absolute) / 2))]
        ln_abs = np.log(absolute)
        return ln_abs

      ## data sicing NS
      ln_abs_NS = fouier(data_new.CBA)
      data_set = pd.DataFrame({"ln_abs_2" : ln_abs_NS})

      data_new = pd.DataFrame({"UTM_X": data_new.UTM_X, "UTM_Y": data_new.UTM_Y, "CBA": data_new.CBA, "Interval": dist})

      ## wavenumber (K)
      def wavenumber(DF_K):
        dt = (len(DF_K.ln_abs_2)*space)
        A = np.arange(1,len(DF_K.ln_abs_2))

        f_sample = []
        for i in A:
          f_s = (i/2)/dt
          f_sample.append(f_s)
        f_sample = np.append(0, f_sample)

        k = 2 * np.pi * f_sample
        return k
      k = wavenumber(data_set)
      data_set = pd.DataFrame({"ln_abs_2" : ln_abs_NS, "k_2" : k})


      plt.plot(data_set.k_2, data_set.ln_abs_2, ".")
      print("panjang:", len(data_set.k_2))
      plt.show()
      # print(data_2)

      def cut_off(reg_cut,res_cut):
        k_reg_cut = k[:reg_cut]; ln_reg_cut = data_set.ln_abs_2[:reg_cut]
        k_res_cut = k[reg_cut:res_cut]; ln_res_cut = data_set.ln_abs_2[reg_cut:res_cut]
        k_noise_cut = k[res_cut:]; ln_noise_cut = data_set.ln_abs_2[res_cut:]

        return (k_reg_cut,ln_reg_cut,k_res_cut,ln_res_cut,k_noise_cut,ln_noise_cut)

      k_reg_cut,ln_reg_cut,k_res_cut,ln_res_cut,k_noise_cut,ln_noise_cut = cut_off(int(input("Masukan nilai cut off regional:")),
      int(input("Masukan nilai cut off residual:")))

      def simple_linear_regression_traditional(x, y):
        from sklearn.linear_model import LinearRegression
        x = np.array(x);y = np.array(y)
        regr = LinearRegression()
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        regr.fit(x, y)
        B0 = regr.intercept_
        B1 = regr.coef_

        return(B0,B1) # B0 intercept, B1 gradient

      #Regional Zone
      B0, B1 = simple_linear_regression_traditional(k_reg_cut, ln_reg_cut)

      #Residual Zone
      B2, B3 = simple_linear_regression_traditional(k_res_cut, ln_res_cut)

      # cut-off
      cut_off = (B2-B0)/(B1-B3)

      #window
      windows = 2*np.pi/(space*cut_off)

      print("Calculated window: ",windows, "\n")
      print("Intercept for Regional Zone: ",B0)
      print("Gradient for Regional Zone: ",B1)
      print("Intercept for Residual Zone: ",B2)
      print("Gradient for Residual Zone: ",B3)
      print("Cut-off frequency: ",cut_off)

      rounded = np.round(windows)
      if rounded % 2 == 0:
        if (rounded - windows) > 0:
          window = rounded - 1
        else :
          window = rounded + 1
      else:
        window = rounded

      # print(window)

      y_reg = B0 + B1 * k_reg_cut
      y_reg = y_reg.flatten()
      y_res = B2 + B3 * k_res_cut
      y_res = y_res.flatten()

      plt.plot(k, data_set.ln_abs_2, '.-')
      p1 = plt.plot(k_reg_cut, ln_reg_cut, '.-', color='red')
      p2 = plt.plot(k_res_cut, ln_res_cut, '.-', color='green')
      p3 = plt.plot(k_noise_cut, ln_noise_cut, '.-', color='orange')
      p4 = plt.plot(k_reg_cut, y_reg, '--', color='red', linewidth=3)
      p5 = plt.plot(k_res_cut, y_res, '--', color='green', linewidth=3)
      plt.legend((p1[0], p2[0], p3[0], p4[0],p5[0]),['Regional', 'Residual', 'Noise','linear regression regional','linear regression residual'])
      plt.title('Spectral Analysis')
      plt.xlabel('Wavenumber (k)'); plt.ylabel('ln(absolute)')
      data_new = pd.DataFrame({"UTM_X": data_new.UTM_X, "UTM_Y": data_new.UTM_Y, "CBA": data_new.CBA, "Interval": dist})
      # print(window)
      plt.show()

      print(500*'=')
      answare2 = input("Do you want to repeat the spectrum analysis process (Y/N)?")
      if answare2 == 'N':
        break

    print("enter window width, for moving average data")
    ##Menghitung rata-rata window
    daftar_input = input('Window width : ')
    list_windows = daftar_input.split(',')
    new_window = [int(x) for x in list_windows]

    jumlah = 0
    for num in new_window:
        jumlah += num
    average = jumlah / len(new_window)
    average = int(average)
    if average % 2 == 0:
      rata_rata = average + 1
    else:
      rata_rata = average
    print(average)
    m, n = average, average  # The shape of the window array
    win = np.ones((m, n))

    def moving_average_2d(data, window):
      """Moving average on 2D data.
      """
      window /= window.sum()
      if type(data).__name__ not in ['ndarray', 'MaskedArray']:
        data = np.asarray(data)
      from scipy.signal import convolve2d
      result = convolve2d(data, window, mode='same', boundary='symm')
      return result, window

    regional, window = moving_average_2d(zi, win)
    residual = zi - regional

    fig, ax_1 = plt.subplots(figsize=(12,10))
    im_2 = ax_1.contourf(xi,yi,regional,levels=40, cmap="jet")
    plt.colorbar(im_2, label = "mGal")
    ax_1.set_title('Regional Anomaly')
    ax_1.set_xlabel('Easthing (m)'); plt.ylabel('Northing (m)')
    # plt.show()
    plt.figure(1)
    fig, ax_2 = plt.subplots(figsize=(12,10))
    im_2 = ax_2.contourf(xi,yi,residual,levels=40, cmap="jet")
    ax_2.set_title('Residual Anomaly')
    ax_2.set_xlabel('Easthing (m)'); plt.ylabel('Northing (m)')
    plt.colorbar(im_2, label = "mGal")
    plt.show()
    answare1 = input("apakah Anda ingin mengulangi seluruh langkah analisis spektrum? (Y/N) :" )
    if answare1 == 'N':
      break
  return regional, residual


