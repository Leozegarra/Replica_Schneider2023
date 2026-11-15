# -*- coding: utf-8 -*-
"""
PASO 3. Comparar nuestros resultados con los del artículo.

Hace dos comprobaciones:
  A) Con NUESTRAS canastas (paso 1, resueltas en Python con HiGHS) recalcula la
     factibilidad y el costo mediano de la olla compartida (Tabla 2).
  B) Junta en una sola tabla: valor del artículo, valor con los datos de la autora
     procesados en Python (paso 2) y valor con nuestras propias canastas.
"""
import numpy as np
import pandas as pd

R = "resultados/"


def media_p(x, w):
    m = x.notna() & w.notna(); return np.average(x[m], weights=w[m])


def mediana_p(x, w):
    m = x.notna() & w.notna()
    x, w = x[m].to_numpy(float), w[m].to_numpy(float)
    o = np.argsort(x); x, w = x[o], w[o]
    return x[np.searchsorted(np.cumsum(w) / w.sum(), 0.5)]


can = pd.read_csv(R + "canastas_resueltas.csv.gz")
can["comp_pc_replica"] = can.CoNA_replica_pd / can.PPP_month_interpolated / can.hhsize
fact_rep = media_p(can.factible_replica.astype(float) * 100, can.pweight)
costo_rep = mediana_p(can.comp_pc_replica, can.pweight)

t2 = pd.read_csv(R + "tabla2.csv", index_col=0)
t3 = pd.read_csv(R + "tabla3.csv", index_col=0)

filas = [
    # indicador, valor reportado en el artículo, réplica con datos de la autora, réplica con nuestras canastas
    ("Hogares rurales analizados", 3117, 3117, can.hhxround.nunique()),
    ("% meses con dieta factible, individualizado", 89.72, t2.iloc[0, 0], None),
    ("% meses con dieta factible, olla compartida", 60.18, t2.iloc[0, 1], fact_rep),
    ("Costo mediano por persona al día, individualizado (US$ PPP)", 1.79, t2.iloc[1, 0], None),
    ("Costo mediano por persona al día, olla compartida (US$ PPP)", 2.26, t2.iloc[1, 1], costo_rep),
    ("% sin acceso sin gastar más en comida, individualizado", 62, t3.iloc[4, 0], None),
    ("% sin acceso sin gastar más en comida, olla compartida", 80, t3.iloc[4, 1], None),
    ("Prima por compartir la olla (%)", 33, t3.iloc[6, 1], None),
]
comp = pd.DataFrame(filas, columns=["Indicador", "Artículo", "Réplica (datos autora, Python)",
                                    "Réplica (nuestras canastas HiGHS)"])
comp.iloc[:, 1:] = comp.iloc[:, 1:].apply(pd.to_numeric).round(2)
comp.to_csv(R + "comparacion_con_paper.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 70)
print(comp.to_string(index=False))
