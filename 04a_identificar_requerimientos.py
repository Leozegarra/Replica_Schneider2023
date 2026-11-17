# -*- coding: utf-8 -*-
"""
PASO 4a. Identificar qué requerimiento individual usó la autora para cada grupo.

En Dataverse los requerimientos vienen por PERSONA (HHNeeds_IndivLevel_r.dta) y varían un
poco dentro de cada grupo de edad y sexo (por ejemplo, la energía depende de la edad en meses).
La autora tomó un solo valor por grupo con "collapse (first)" de Stata, que depende del orden
interno de los datos y no se puede reconstruir exactamente.

Por eso probamos, para cada grupo, todos los vectores de requerimientos distintos que existen
(datos/candidatos_requerimientos.csv.gz, 378 en total) en una muestra de 100 mercados-mes, y nos
quedamos con el que mejor reproduce la factibilidad y el costo publicados por la autora.
Salida: datos/requerimientos_individuales.csv.gz (lo usa el paso 4).
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

NUTRIENTES = ["energy", "carbohydrate", "protein", "lipids", "vitA", "retinol", "vitC", "vitE",
              "thiamin", "riboflavin", "niacin", "vitB6", "folate", "vitB12", "calcium", "copper",
              "iron", "magnesium", "phosphorus", "selenium", "zinc", "sodium"]
V = ["ear_perweek", "ul_perweek", "amdr_lower_perweek", "amdr_upper_perweek"]

cand = pd.read_csv("datos/candidatos_requerimientos.csv.gz")
precios = pd.read_csv("datos/precios_mercados.csv.gz")
autora = pd.read_csv("datos/icona_autora.csv.gz")
mats = {}
for k, d in precios.groupby(["market_no", "year", "month"]):
    d = d[d.uni_price_net.notna() & (d.uni_price_net > 0)]
    mats[k] = (d.uni_price_net.values / 1000, d[NUTRIENTES].fillna(0).values.T / 100)


def vector(fila, var):
    return np.array([fila.get(f"{var}_{k}", np.nan) for k in range(2, 24)], float)


def resolver(c, A, E, U, L, H):
    f, b = [], []
    for k in range(1, 22):
        for v, s in ((E[k], -1), (L[k], -1), (U[k], 1), (H[k], 1)):
            if not np.isnan(v) and v != 0:
                f.append(s * A[k]); b.append(s * v)
    r = linprog(c, A_ub=np.array(f), b_ub=np.array(b), A_eq=A[:1], b_eq=[E[0]], bounds=(0, None), method="highs")
    return r.fun if r.status == 0 else np.nan


elegidos, resumen = [], []
for (g, pal), cg in cand.groupby(["age_sex_grp", "pal"]):
    muestra = autora[(autora.age_sex_grp == g) & (autora.pal == pal)].sample(100, random_state=1)
    mejor = None
    for _, fila in cg.iterrows():
        f = fila.to_dict()
        E, U, L, H = (vector(f, v) for v in V)
        for k in (3, 5, 21):                      # carbohidratos, grasas y sodio sin EAR, como la autora
            if E[k] == 0: E[k] = np.nan
        coincide, errores = 0, []
        for _, x in muestra.iterrows():
            v = resolver(*mats[(x.market_no, x.year, x.month)], E, U, L, H)
            coincide += np.isnan(v) == np.isnan(x.iCoNA)
            if not (np.isnan(v) or np.isnan(x.iCoNA)):
                errores.append(abs(v / x.iCoNA - 1))
        puntaje = (coincide, -np.median(errores) if errores else -9)
        if mejor is None or puntaje > mejor[0]:
            mejor = (puntaje, f)
    (coin, err), f = mejor
    resumen.append((int(g), int(pal), len(cg), coin, -err))
    for k in range(1, 24):
        elegidos.append({"age_sex_grp": g, "pal": pal, "nutr_no": k, **{v: f.get(f"{v}_{k}", np.nan) for v in V}})
    print(f"grupo {int(g):>2} pal {int(pal)}: {len(cg):>2} candidatos, coincide factibilidad {coin}/100, "
          f"error mediano {-err:.2e}", flush=True)

pd.DataFrame(elegidos).to_csv("datos/requerimientos_individuales.csv.gz", index=False)
pd.DataFrame(resumen, columns=["grupo", "pal", "candidatos", "coincide_de_100", "error_mediano"]).to_csv(
    "resultados/identificacion_requerimientos.csv", index=False)
