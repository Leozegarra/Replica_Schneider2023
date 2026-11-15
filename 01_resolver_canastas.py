# -*- coding: utf-8 -*-
"""
PASO 1. Volver a resolver, con nuestro propio código, la canasta de menor costo
de cada hogar en cada mes (escenario "compartido" del artículo: CoNA_sharing).

Idea del artículo (Schneider et al., 2023, sección de métodos):
  - Cada hogar come de una sola olla. Para no dejar a nadie por debajo de lo que
    necesita, los mínimos del hogar se fijan con el miembro "más exigente" por
    caloría (max EAR_i/E_i) y los máximos con el más sensible (min UL_i/E_i).
  - Con esos requerimientos se arma un problema de programación lineal:
        minimizar   suma( precio_j * gramos_j )
        sujeto a    energía del hogar exacta (igualdad)
                    nutrientes >= EAR      (no quedarse corto)
                    nutrientes <= UL       (no pasarse a niveles dañinos)
                    macronutrientes dentro del rango AMDR
                    gramos_j >= 0
  - Se resuelve con los precios del mercado más cercano en cada mes.
  - Si no existe solución, la dieta es "no factible" en ese mercado y mes.

La autora lo resolvió en R con lpSolve. Aquí usamos scipy (solver HiGHS),
que es otro solver, para comprobar que el resultado no depende de la herramienta.

Salida: resultados/canastas_resueltas.csv.gz  y un resumen en pantalla.
"""
import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog

NUTRIENTES = ["energy", "carbohydrate", "protein", "lipids", "vitA", "retinol", "vitC", "vitE",
              "thiamin", "riboflavin", "niacin", "vitB6", "folate", "vitB12", "calcium", "copper",
              "iron", "magnesium", "phosphorus", "selenium", "zinc", "sodium"]  # nutr_no 2..23

# ---------------------------------------------------------------- datos
serie = pd.read_csv("datos/serie_mensual.csv.gz", parse_dates=["dateMY"], low_memory=False)
req = pd.read_csv("datos/requerimientos_hogar.csv.gz").drop(columns=["market_no", "data_round", "hhsize"])
precios = pd.read_csv("datos/precios_mercados.csv.gz")

# Misma muestra que el artículo: zona rural, rondas 2013 y 2016-17 (data_round 2 y 3),
# hogares con mercado asignado.
serie = serie[(serie.reside == 1) & (serie.data_round != 1) & serie.market_no.notna()].copy()
serie["anio"] = serie.dateMY.dt.year
serie["mes"] = serie.dateMY.dt.month
serie = serie.merge(req, on="hhxround", how="left")

# --------------------------------------------------- matrices por mercado y mes
# Precio en kwachas por kg -> por gramo. Nutrientes por 100 g -> por gramo.
# Un alimento sin precio ese mes no se puede comprar (se excluye).
matrices = {}
for clave, g in precios.groupby(["market_no", "year", "month"]):
    g = g[g.uni_price_net.notna() & (g.uni_price_net > 0)]
    matrices[clave] = (g.uni_price_net.values / 1000.0, g[NUTRIENTES].fillna(0).values.T / 100.0)

# Requerimientos en arreglos (filas = hogar-mes, columnas = nutriente 2..23)
def arreglo(prefijo):
    return serie[[f"{prefijo}_sharing_{k}" for k in range(2, 24)]].to_numpy(float)
EAR, UL, LOW, UP = arreglo("EAR"), arreglo("UL"), arreglo("AMDRlow"), arreglo("AMDRup")


def resolver(i, costo, A):
    """Arma y resuelve el PL de un hogar-mes. Devuelve (costo semanal, gramos)."""
    filas, lim = [], []
    for k in range(1, 22):  # k=0 es la energía (va como igualdad)
        for valor, signo in ((EAR[i, k], -1), (LOW[i, k], -1), (UL[i, k], 1), (UP[i, k], 1)):
            if not np.isnan(valor) and valor != 0:
                filas.append(signo * A[k]); lim.append(signo * valor)
    r = linprog(costo, A_ub=np.array(filas), b_ub=np.array(lim),
                A_eq=A[:1], b_eq=[EAR[i, 0]], bounds=(0, None), method="highs")
    return (r.fun, r.x) if r.status == 0 else (np.nan, None)


inicio = time.time()
costos = np.full(len(serie), np.nan)
claves = list(zip(serie.market_no, serie.anio, serie.mes))
for i, clave in enumerate(claves):
    if clave in matrices and not np.isnan(EAR[i, 0]):
        costos[i] = resolver(i, *matrices[clave])[0]
    if i % 10000 == 0:
        print(f"  {i:>6} de {len(serie)} hogar-mes resueltos ({time.time() - inicio:.0f} s)")

serie["CoNA_replica"] = costos                  # costo por semana del hogar, en kwachas
serie["CoNA_replica_pd"] = costos / 7           # por día
serie["factible_replica"] = ~np.isnan(costos)
print(f"Tiempo total: {time.time() - inicio:.0f} s para {len(serie)} problemas")

# --------------------------------------------------------- comparación
autora = serie.CoNA_sharing.notna()
nuestro = serie.factible_replica
ambos = autora & nuestro
dif = serie.loc[ambos, "CoNA_replica"] / serie.loc[ambos, "CoNA_sharing"] - 1
print(f"Factibles autora: {autora.mean():.2%}   Factibles réplica: {nuestro.mean():.2%}")
print(f"Coincide la factibilidad en el {(autora == nuestro).mean():.2%} de los casos")
print(f"Diferencia relativa de costo: mediana {dif.median():.2e}, máxima {dif.abs().max():.2e}")
print(f"Correlación costos autora vs réplica: {np.corrcoef(serie.loc[ambos,'CoNA_replica'], serie.loc[ambos,'CoNA_sharing'])[0,1]:.8f}")

serie[["hhxround", "dateMY", "market_no", "data_round", "pweight", "hhsize", "PPP_month_interpolated",
       "CoNA_sharing", "CoNA_replica", "CoNA_replica_pd", "factible_replica"]].to_csv(
    "resultados/canastas_resueltas.csv.gz", index=False)
