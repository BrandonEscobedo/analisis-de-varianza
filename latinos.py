import numpy as np
import pandas as pd
from collections import defaultdict
p = 2
r = 10
REPLICA_1 = np.array([
    [311, 265],
    [330, 350]
])

REPLICA_2 = np.array([
    [305, 351],
    [276, 220]
])

REPLICA_3 = np.array([
    [319, 264],
    [254, 307]
])

REPLICA_4 = np.array([
    [159, 205],
    [262, 246]
])

REPLICA_5 = np.array([
    [283, 250],
    [237, 300]
])

REPLICA_6 = np.array([
    [290, 210],
    [232, 299]
])

REPLICA_7 = np.array([
    [248, 318],
    [304, 260]
])

REPLICA_8 = np.array([
    [250, 297],
    [342, 251]
])

REPLICA_9 = np.array([
    [341, 260],
    [210, 333]
])

REPLICA_10 = np.array([
    [218, 299],
    [250, 257]
])
REPLICA_LATINO_1 = np.array([
    ['A', 311, 'B', 265],
    ['B', 330, 'A', 350]
])

REPLICA_LATINO_2 = np.array([
    ['B', 305, 'A', 351],
    ['A', 276, 'B', 220]
])

REPLICA_LATINO_3 = np.array([
    ['A', 319, 'B', 264],
    ['B', 254, 'A', 307]
])

REPLICA_LATINO_4 = np.array([
    ['B', 159, 'A', 205],
    ['A', 262, 'B', 246]
])

REPLICA_LATINO_5 = np.array([
    ['A', 283, 'B', 250],
    ['B', 237, 'A', 300]
])

REPLICA_LATINO_6 = np.array([
    ['A', 290, 'B', 210],
    ['B', 232, 'A', 299]
])

REPLICA_LATINO_7 = np.array([
    ['B', 248, 'A', 318],
    ['A', 304, 'B', 260]
])

REPLICA_LATINO_8 = np.array([
    ['B', 250, 'A', 297],
    ['A', 342, 'B', 251]
])

REPLICA_LATINO_9 = np.array([
    ['A', 341, 'B', 260],
    ['B', 210, 'A', 333]
])

REPLICA_LATINO_10 = np.array([
    ['B', 218, 'A', 299],
    ['A', 250, 'B', 257]
])
replicas_latino = [REPLICA_1, REPLICA_2, REPLICA_3, REPLICA_4, REPLICA_5, REPLICA_6, REPLICA_7, REPLICA_8, REPLICA_9, REPLICA_10]
replicas_letras = [REPLICA_LATINO_1, REPLICA_LATINO_2, REPLICA_LATINO_3, REPLICA_LATINO_4, REPLICA_LATINO_5, REPLICA_LATINO_6, REPLICA_LATINO_7, REPLICA_LATINO_8, REPLICA_LATINO_9, REPLICA_LATINO_10]

tratamientos = ['A', 'B',]

df_replicas = pd.DataFrame(np.array(replicas_latino).reshape(r * p, p), columns=[f'P{i+1}' for i in range(p)])

sumas_tratamientos = {tratamiento: 0 for tratamiento in tratamientos}
for replica in replicas_latino:
    for fila in replica:
        for idx, tratamiento in enumerate(tratamientos):
            sumas_tratamientos[tratamiento] += fila[idx]

total_suma = sum(sumas_tratamientos.values())

N = r * p * p  
r_p=r*p
gl_tratamiento = p - 1
gl_renglones = p - 1
gl_columnas = p - 1
gl_replicas = r - 1
gl_error = (p - 1) * (r * (p + 1) - 3)
gl_total = N - 1

sc_total = np.sum([np.sum(replica**2) for replica in replicas_latino]) - (total_suma**2 / N)

sc_tratamiento = 0
suma_total = defaultdict(int)
for replica in replicas_letras:
    for fila in replica:
        for i in range(0, len(fila), 2):
            letra = fila[i]
            valor = int(fila[i + 1])
            suma_total[letra] += valor

total_letras = [suma_total[letra] for letra in sorted(suma_total.keys())]
for x in total_letras:
    sc_tratamiento += (x ** 2) / r_p
sc_tratamiento -= (total_suma**2 / N)

sc_replicas = np.sum([np.sum(replica)**2 for replica in replicas_latino]) / (p * p) - (total_suma**2 / N)

yi = np.sum([np.sum(replica, axis=1) for replica in replicas_latino], axis=0)
sumas_cuadrad_yi = (yi**2) /r_p
sc_renglones = np.sum(sumas_cuadrad_yi) - (total_suma**2 / N)

yk = np.sum([np.sum(replica, axis=0) for replica in replicas_latino], axis=0)
sumas_cuadrad_yk = (yk**2) / r_p

sc_columnas = np.sum(sumas_cuadrad_yk) - (total_suma**2 / N)

sc_error = sc_total - sc_renglones - sc_tratamiento - sc_columnas - sc_replicas

cm_tratamiento = sc_tratamiento / gl_tratamiento
cm_renglones = sc_renglones / gl_renglones
cm_columnas = sc_columnas / gl_columnas
cm_replicas = sc_replicas / gl_replicas
cm_error = sc_error / gl_error

f_tratamiento = cm_tratamiento / cm_error
f_renglones = cm_renglones / cm_error
f_columnas = cm_columnas / cm_error
f_replicas = cm_replicas / cm_error

anova_df = pd.DataFrame({
    "Fuente de Variación": ["Renglones", "Tratamientos latinos", "Columnas", "Réplicas", "Error", "Total"],
    "Grados de libertad": [gl_renglones , gl_tratamiento, gl_columnas, gl_replicas, gl_error, gl_total],
    "Suma de cuadrados": [sc_renglones, sc_tratamiento, sc_columnas, sc_replicas, sc_error, sc_total],
    "Cuadrado medio": [cm_renglones, cm_tratamiento, cm_columnas, cm_replicas, cm_error, None],
    "F": [f_renglones, f_tratamiento, f_columnas, f_replicas, None, None]
})

print(anova_df)
