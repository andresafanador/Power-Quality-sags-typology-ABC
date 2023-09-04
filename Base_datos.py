import numpy as np
from numpy import linalg as LA
import pandas as pd
import random as rd
import cmath as cm
import h5py as h5

def random_data(vec_r):
    vec1=vec_r[0]
    vec2=vec_r[1]
    vec3=vec_r[2]
    num=rd.randint(0,2)
    if num==0:
        return [vec1,vec2,vec3]
    if num==1:
        return [vec3,vec1,vec2]
    if num==2:
        return [vec2,vec3,vec1]


def u(t):
    """
    Definición de la función escalar unitaria.

    :param t:
    :return: 0 x 1
    """
    return np.piecewise(t, [t < 0.0, t >= 0.0], [0, 1])


def typeA(cantidad_datos, time=False):
    """
    Genera ondas sin ningun problema a 60 Hz

    """
    x_n = []

    t = np.linspace(0, 1 / 6, int(1 / 6 * 2000))
    #   ///////datos necesarios para el df/////

    # guardamos los datos en el df

    for i in range(0, cantidad_datos):
        x = []

        e1 = rd.uniform(0.1, 0.69)
        e2 = e1 + rd.uniform(-e1 / 10, e1 / 10)
        e3 = e1 + rd.uniform(-e1 / 10, e1 / 10)

        ua = e1
        ub = -(e1 + e2 * 1.732j) / 2
        uc = -(e1 - e3 * 1.732j) / 2
        # caracteristicas sag y de la onda
        desfase = rd.uniform(0, 2 * np.pi)  # desfase de la onda
        duracion = rd.uniform(1 / 60, 1 / 5)
        alpha1 = 1 - abs(ua)
        alpha2 = 1 - abs(ub)
        alpha3 = 1 - abs(uc)
        duracion = rd.uniform(1 / 60, 1 / 5)
        t_1 = rd.uniform(0, 1 / 7.5 - duracion)
        t_2 = t_1 + duracion  # el valor final depende de la duración
        # resultado: onda con sag y guardamos en el dict
        f_n1 = (1 - alpha1 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(ua))
        f_n2 = (1 - alpha2 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(ub))
        f_n3 = (1 - alpha3 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(uc))
        x = [list(f_n1), list(f_n2), list(f_n3)]
        x_n.append(x)
    # Se guarda el dict como la base de datos, primera columna es el vector t

    # Exportamos csv
    if time == False:
        return x_n
    else:
        return x_n, list(t)


def typeB(cantidad_datos, time=False):
    """
    Genera ondas sin ningun problema a 60 Hz

    """
    x_n = []

    t = np.linspace(0, 1 / 6, int(1 / 6 * 2000))
    #   ///////datos necesarios para el df/////

    # guardamos los datos en el df

    for i in range(0, cantidad_datos):
        x = []
        e1 = rd.uniform(0.9, 1.1)
        e2 = rd.uniform(0.9, 1.1)
        e3 = rd.uniform(0.9, 1.1)

        ua = rd.uniform(0.1, 0.69)
        ub = -(e1 + e2 * 1.732j) / 2
        uc = -(e1 - e3 * 1.732j) / 2
        # caracteristicas sag y de la onda
        desfase = rd.uniform(0, 2 * np.pi)  # desfase de la onda
        duracion = rd.uniform(1 / 60, 1 / 5)
        alpha1 = 1 - abs(ua)
        alpha2 = 1 - abs(ub)
        alpha3 = 1 - abs(uc)
        duracion = rd.uniform(1 / 60, 1 / 5)
        t_1 = rd.uniform(0, 1 / 7.5 - duracion)
        t_2 = t_1 + duracion  # el valor final depende de la duración
        # resultado: onda con sag y guardamos en el dict
        f_n1 = (1 - alpha1 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(ua))
        f_n2 = (1 - alpha2 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(ub))
        f_n3 = (1 - alpha3 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(uc))
        x = [list(f_n1), list(f_n2), list(f_n3)]
        x_n.append(x)
    # Se guarda el dict como la base de datos, primera columna es el vector t

    # Exportamos csv
    if time == False:
        return x_n
    else:
        return x_n, list(t)


def typeC(cantidad_datos, time=False):
    """
    Genera ondas sin ningun problema a 60 Hz

    """
    x_n = []

    t = np.linspace(0, 1 / 6, int(1 / 6 * 2000))
    #   ///////datos necesarios para el df/////

    # guardamos los datos en el df

    for i in range(0, cantidad_datos):
        x = []
        e1 = rd.uniform(0.9, 1.1)
        e2 = rd.uniform(0.1, 0.69)
        e3 = e2 + rd.uniform(-e2 / 10, e2 / 10)

        ua = e1
        ub = -(e1 + e2 * 1.732j) / 2
        uc = -(e1 - e3 * 1.732j) / 2
        # caracteristicas sag y de la onda
        desfase = rd.uniform(0, 2 * np.pi)  # desfase de la onda
        duracion = rd.uniform(1 / 60, 1 / 5)
        alpha1 = 1 - abs(ua)
        alpha2 = 1 - abs(ub)
        alpha3 = 1 - abs(uc)
        duracion = rd.uniform(1 / 60, 1 / 5)
        t_1 = rd.uniform(0, 1 / 7.5 - duracion)
        t_2 = t_1 + duracion  # el valor final depende de la duración
        # resultado: onda con sag y guardamos en el dict
        f_n1 = (1 - alpha1 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(ua))
        f_n2 = (1 - alpha2 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(ub))
        f_n3 = (1 - alpha3 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(uc))
        x = [list(f_n1), list(f_n2), list(f_n3)]
        x_n.append(x)
    # Se guarda el dict como la base de datos, primera columna es el vector t

    # Exportamos csv
    if time == False:
        return x_n
    else:
        return x_n, list(t)


def typeD(cantidad_datos, time=False):
    """
    Genera ondas sin ningun problema a 60 Hz

    """

    x_n = []
    t = np.linspace(0, 1 / 6, int(1 / 6 * 2000))
    #   ///////datos necesarios para el df/////

    # guardamos los datos en el df

    for i in range(0, cantidad_datos):
        x = []
        e1 = rd.uniform(0.1, 0.69)
        e2 = rd.uniform(0.9, 1.1)
        e3 = rd.uniform(0.9, 1.1)

        ua = e1
        ub = -(e1 + e2 * 1.732j) / 2
        uc = -(e1 - e3 * 1.732j) / 2
        # caracteristicas sag y de la onda
        desfase = rd.uniform(0, 2 * np.pi)  # desfase de la onda
        duracion = rd.uniform(1 / 60, 1 / 5)
        alpha1 = 1 - abs(ua)
        alpha2 = 1 - abs(ub)
        alpha3 = 1 - abs(uc)
        duracion = rd.uniform(1 / 60, 1 / 5)
        t_1 = rd.uniform(0, 1 / 7.5 - duracion)
        t_2 = t_1 + duracion  # el valor final depende de la duración
        # resultado: onda con sag y guardamos en el dict
        f_n1 = (1 - alpha1 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(ua))
        f_n2 = (1 - alpha2 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(ub))
        f_n3 = (1 - alpha3 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(uc))
        x = [list(f_n1), list(f_n2), list(f_n3)]
        x_n.append(x)
    # Se guarda el dict como la base de datos, primera columna es el vector t

    # Exportamos csv
    if time == False:
        return x_n
    else:
        return x_n, list(t)


def typeE(cantidad_datos, time=False):
    """
    Genera ondas sin ningun problema a 60 Hz

    """

    x_n = []
    t = np.linspace(0, 1 / 6, int(1 / 6 * 2000))
    #   ///////datos necesarios para el df/////

    # guardamos los datos en el df

    for i in range(0, cantidad_datos):
        x = []
        e1 = rd.uniform(0.1, 0.69)
        e2 = e1 + rd.uniform(-e1 / 10, e1 / 10)
        e3 = e1 + rd.uniform(-e1 / 10, e1 / 10)

        ua = rd.uniform(0.9, 1.1)
        ub = -(e1 + e2 * 1.732j) / 2
        uc = -(e1 - e3 * 1.732j) / 2
        # caracteristicas sag y de la onda
        desfase = rd.uniform(0, 2 * np.pi)  # desfase de la onda
        duracion = rd.uniform(1 / 60, 1 / 5)
        alpha1 = 1 - abs(ua)
        alpha2 = 1 - abs(ub)
        alpha3 = 1 - abs(uc)
        duracion = rd.uniform(1 / 60, 1 / 5)
        t_1 = rd.uniform(0, 1 / 7.5 - duracion)
        t_2 = t_1 + duracion  # el valor final depende de la duración
        # resultado: onda con sag y guardamos en el dict
        f_n1 = (1 - alpha1 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(ua))
        f_n2 = (1 - alpha2 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(ub))
        f_n3 = (1 - alpha3 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(uc))
        x = [list(f_n1), list(f_n2), list(f_n3)]
        x_n.append(x)
    # Se guarda el dict como la base de datos, primera columna es el vector t

    # Exportamos csv
    if time == False:
        return x_n
    else:
        return x_n, list(t)


def typeF(cantidad_datos, time=False):
    """
    Genera ondas sin ningun problema a 60 Hz

    """

    x_n = []
    t = np.linspace(0, 1 / 6, int(1 / 6 * 2000))
    #   ///////datos necesarios para el df/////

    # guardamos los datos en el df

    for i in range(0, cantidad_datos):
        x = []
        e1 = rd.uniform(0.1, 0.69)
        e2 = e1 + rd.uniform(-e1 / 10, e1 / 10)
        e3 = e1 + rd.uniform(-e1 / 10, e1 / 10)
        E = rd.uniform(0.9, 1.1)
        ua = e1
        ub = (-e1 / 2) - ((E / 3) + (e2 / 6)) * 1.732j
        uc = (-e1 / 2) + ((E / 3) + (e3 / 6)) * 1.732j
        # caracteristicas sag y de la onda
        desfase = rd.uniform(0, 2 * np.pi)  # desfase de la onda
        duracion = rd.uniform(1 / 60, 1 / 5)
        alpha1 = 1 - abs(ua)
        alpha2 = 1 - abs(ub)
        alpha3 = 1 - abs(uc)
        duracion = rd.uniform(1 / 60, 1 / 5)
        t_1 = rd.uniform(0, 1 / 7.5 - duracion)
        t_2 = t_1 + duracion  # el valor final depende de la duración
        # resultado: onda con sag y guardamos en el dict
        f_n1 = (1 - alpha1 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(ua))
        f_n2 = (1 - alpha2 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(ub))
        f_n3 = (1 - alpha3 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(uc))
        x = [list(f_n1), list(f_n2), list(f_n3)]
        x_n.append(x)
    # Se guarda el dict como la base de datos, primera columna es el vector t

    # Exportamos csv
    if time == False:
        return x_n
    else:
        return x_n, list(t)


def typeG(cantidad_datos, time=False):
    """
      Genera ondas sin ningun problema a 60 Hz

      """

    x_n = []
    t = np.linspace(0, 1 / 6, int(1 / 6 * 2000))
    #   ///////datos necesarios para el df/////

    # guardamos los datos en el df

    for i in range(0, cantidad_datos):
        x = []
        e1 = rd.uniform(0.1, 0.79)
        e2 = e1 + rd.uniform(-e1 / 10, e1 / 10)
        e3 = e1 + rd.uniform(-e1 / 10, e1 / 10)
        E = rd.uniform(0.9, 1.1)
        ua = (E * 2 / 3) + (e1 / 3)
        ub = -((E / 3) + (e1 / 6)) - (e2 / 2) * 1.732j
        uc = -((E / 3) + (e1 / 6)) + (e3 / 2) * 1.732j
        # caracteristicas sag y de la onda
        desfase = rd.uniform(0, 2 * np.pi)  # desfase de la onda
        duracion = rd.uniform(1 / 60, 1 / 5)
        alpha1 = 1 - abs(ua)
        alpha2 = 1 - abs(ub)
        alpha3 = 1 - abs(uc)
        duracion = rd.uniform(1 / 60, 1 / 5)
        t_1 = rd.uniform(0, 1 / 7.5 - duracion)
        t_2 = t_1 + duracion  # el valor final depende de la duración
        # resultado: onda con sag y guardamos en el dict
        f_n1 = (1 - alpha1 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(ua))
        f_n2 = (1 - alpha2 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(ub))
        f_n3 = (1 - alpha3 * (u(t - t_1) - u(t - t_2))) * np.sin(2 * np.pi * 60 * t + desfase + cm.phase(uc))
        x = [list(f_n1), list(f_n2), list(f_n3)]
        x_n.append(x)
    # Se guarda el dict como la base de datos, primera columna es el vector t

    # Exportamos csv
    if time == False:
        return x_n
    else:
        return x_n, list(t)
def typeNormal(cantidad_datos, time=False):
    """
      Genera ondas sin ningun problema a 60 Hz

      """

    x_n = []
    t = np.linspace(0, 1 / 6, int(1 / 6 * 2000))
    #   ///////datos necesarios para el df/////

    # guardamos los datos en el df
    
    for i in range(0, cantidad_datos):
        x = []
        a1=rd.uniform(0.95, 1.05)
        a2=rd.uniform(0.95, 1.05)
        a3=rd.uniform(0.95, 1.05)
        desfase = rd.uniform(0, 2 * np.pi)
        desfase2 = rd.uniform(-np.pi/36, np.pi/36)+desfase
        desfase3 = rd.uniform(-np.pi/36, np.pi/36)+desfase
        # resultado: onda con sag y guardamos en el dict
        f_n1 =  a1*np.sin(2 * np.pi * 60 * t + desfase )
        f_n2 = a2*np.sin(2 * np.pi * 60 * t + desfase2 -2.094)
        f_n3 = a3*np.sin(2 * np.pi * 60 * t + desfase3 +2.094 )
        x = [list(f_n1), list(f_n2), list(f_n3)]
        x_n.append(x)
    # Se guarda el dict como la base de datos, primera columna es el vector t

    # Exportamos csv
    if time == False:
        return x_n
    else:
        return x_n, list(t)

def Data_base(numb):
    """
    Organizaremos la base de datos y crearemos un vector para posteriormente
    poder clasificar esta base de datos como :
    [1,0,0,0,0,0,0] ----> A
    [0,1,0,0,0,0,0] ----> B
    [0,0,1,0,0,0,0] ----> C
    [0,0,0,1,0,0,0] ----> D
    [0,0,0,0,1,0,0] ----> E
    [0,0,0,0,0,1,0] ----> F
    [0,0,0,0,0,0,1] ----> G
    """
    x = []
    y = []
    for i in range(numb):
        x.append(random_data(typeA(1)[0]))
        x.append(random_data(typeB(1)[0]))
        x.append(random_data(typeC(1)[0]))
        x.append(random_data(typeD(1)[0]))
        x.append(random_data(typeE(1)[0]))
        x.append(random_data(typeF(1)[0]))
        x.append(random_data(typeG(1)[0]))
        x.append(random_data(typeNormal(1)[0]))
        y.append([1, 0, 0, 0, 0, 0, 0,0])
        y.append([0, 1, 0, 0, 0, 0, 0,0])
        y.append([0, 0, 1, 0, 0, 0, 0,0])
        y.append([0, 0, 0, 1, 0, 0, 0,0])
        y.append([0, 0, 0, 0, 1, 0, 0,0])
        y.append([0, 0, 0, 0, 0, 1, 0,0])
        y.append([0, 0, 0, 0, 0, 0, 1,0])
        y.append([0, 0, 0, 0, 0, 0, 0,1])
    
    X = np.array(x)
    Y = np.array(y)
    X = np.swapaxes(X,0,2)
    with h5.File("Database.h5", "w") as hf:
        hf.create_dataset("X_train", data = X)
        hf.create_dataset("Y_train", data = Y)
     
if __name__ == "__main__":
    Data_base(20000)

