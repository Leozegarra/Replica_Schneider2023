# -*- coding: utf-8 -*-
"""
PASO 3 (correr al final). Comparar nuestros resultados con los del artículo.
Valores del artículo tomados de sus Tablas 1, 2 y 3 (versión arXiv 2207.07240 y AJAE 2023).
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
can["pc"] = can.CoNA_replica_pd / can.PPP_month_interpolated / can.hhsize
t1 = pd.read_csv(R + "tabla1.csv", index_col=0)
t2 = pd.read_csv(R + "tabla2.csv", index_col=0)
t3 = pd.read_csv(R + "tabla3.csv", index_col=0)
ti = pd.read_csv(R + "tabla2_individual_replica.csv", index_col=0)
br = pd.read_csv(R + "brechas_estacionales.csv", index_col=0)["valor"]

filas = [
    # sección, indicador, artículo, réplica con datos de la autora, réplica con nuestro propio solver
    ("Tabla 1", "Hogares rurales", 3117, t1.loc["Número de hogares", "Total"], None),
    ("Tabla 1", "Personas", 14902, t1.loc["Número de personas (sin bebés <6 meses)", "Total"], None),
    ("Tabla 1", "Tamaño del hogar", 4.90, t1.loc["Tamaño del hogar (media)", "Total"], None),
    ("Tabla 1", "Adultos por hogar", 2.44, t1.loc["Adultos >18 (media por persona)", "Total"], None),
    ("Tabla 1", "Niños por hogar", 3.51, t1.loc["Niños <=18 (media por persona)", "Total"], None),
    ("Tabla 1", "Gasto en comida por persona al día (US$)", 1.47, t1.iloc[4, 2], None),
    ("Tabla 1", "Proporción del gasto en comida", 0.74, t1.loc["Proporción del gasto que va a comida (mediana)", "Total"], None),
    ("Tabla 2", "% meses con dieta factible, individual", 89.7, t2.iloc[0, 0], ti.iloc[0, 1]),
    ("Tabla 2", "% meses con dieta factible, compartida", 60.2, t2.iloc[0, 1], media_p(can.factible_replica.astype(float) * 100, can.pweight)),
    ("Tabla 2", "Costo por persona al día, individual (US$)", 1.79, t2.iloc[1, 0], ti.iloc[1, 1]),
    ("Tabla 2", "Costo por persona al día, compartida (US$)", 2.26, t2.iloc[1, 1], mediana_p(can.pc, can.pweight)),
    ("Tabla 2", "Brecha estacional de factibilidad, individual (pp)", 7.4, br["Brecha prob. dieta individual (pp)"], None),
    ("Tabla 2", "Brecha estacional de factibilidad, compartida (pp)", 19.2, br["Brecha prob. dieta compartida (pp)"], None),
    ("Tabla 2", "Brecha estacional de costo, individual (%)", 13.8, br["Brecha estacional costo individual (pp)"], None),
    ("Tabla 2", "Brecha estacional de costo, compartida (%)", 10.1, br["Brecha estacional costo compartido (pp)"], None),
    ("Tabla 3", "% con dieta factible en el mes de encuesta, individual", 86.73, t3.iloc[0, 0], None),
    ("Tabla 3", "% con dieta factible en el mes de encuesta, compartida", 56.13, t3.iloc[0, 1], None),
    ("Tabla 3", "Costo / gasto en comida, individual", 1.11, t3.iloc[2, 0], None),
    ("Tabla 3", "Costo / gasto en comida, compartida", 1.35, t3.iloc[2, 1], None),
    ("Tabla 3", "Costo / gasto total, individual", 0.78, t3.iloc[3, 0], None),
    ("Tabla 3", "Costo / gasto total, compartida", 0.92, t3.iloc[3, 1], None),
    ("Texto", "% sin acceso sin gastar más en comida, individual", 62, t3.iloc[4, 0], None),
    ("Texto", "% sin acceso sin gastar más en comida, compartida", 80, t3.iloc[4, 1], None),
    ("Tabla 3", "Sobrecosto por compartir la olla (%)", 33, t3.iloc[6, 1], None),
]
comp = pd.DataFrame(filas, columns=["Sección", "Indicador", "Artículo", "Réplica (datos de la autora)",
                                    "Réplica (nuestro solver)"])
comp.iloc[:, 2:] = comp.iloc[:, 2:].apply(pd.to_numeric).round(2)
comp.to_csv(R + "comparacion_con_paper.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
print(comp.to_string(index=False))
