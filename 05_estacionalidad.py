# -*- coding: utf-8 -*-
"""
PASO 5. Estacionalidad: en qué meses comer bien cuesta más o no se puede (Tabla 2 y Figura 4).

Sigue la sección "SEASONALITY" del código Stata de la autora:
  1) Costo nominal por persona y su logaritmo, para cada hogar y mes.
  2) Cambio mensual del log del costo (si el mes anterior no tuvo dieta, se usa el último mes
     que sí tuvo; k = cuántos meses atrás se fue).
  3) Regresión ponderada del cambio mensual sobre variables de mes (modelo de tendencia
     estocástica con variables estacionales). El factor estacional de cada mes es su
     coeficiente menos el promedio de los 12; la brecha estacional es el mayor menos el menor.
  4) Lo mismo con la factibilidad (1 = hay dieta ese mes).
  5) Probabilidad de tener dieta por mes del año (ecuación 9 del artículo).
  6) Modelo trigonométrico alternativo y comparación de ajuste (Tabla C del suplemento).
Salidas: resultados/tabla2_estacionalidad.csv, resultados/figura4_estacionalidad.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LLAVES = ["case_id", "HHID", "y2_hhid", "y3_hhid", "data_round"]
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Set", "Oct", "Nov", "Dic"]

s = pd.read_csv("datos/serie_mensual.csv.gz", low_memory=False, parse_dates=["dateMY"])
s = s[(s.year >= 2013) & (s.reside == 1)]
hog = pd.read_csv("datos/hogares.csv.gz")[LLAVES].drop_duplicates()
s = s.merge(hog, on=LLAVES, how="inner")                      # igual que su merge con _merge==3
s = s.sort_values(["hhxround", "dateMY"]).reset_index(drop=True)
s["month"] = s.dateMY.dt.month
s["log_i"] = np.log(s.iCoNA_HH_pd / s.hhsize)               # costo nominal por persona, individual
s["log_s"] = np.log(s.CoNA_sharing_pd / s.hhsize)           # costo nominal por persona, compartido


def diferencia(col):
    """Cambio del log del costo frente al último mes con dato (hasta 12 meses atrás).
    Copia la regla de la autora tal cual. Ojo: su código intenta excluir los primeros meses
    de 2016 con "dateMY!=2016-01", pero en Stata 2016-01 es la resta 2015 y no una fecha,
    así que esa exclusión nunca se aplica. Para reproducir sus cifras tampoco la aplicamos."""
    d = pd.Series(np.nan, index=s.index); k = pd.Series(np.nan, index=s.index)
    for L in range(1, 13):
        cand = s[col] - s[col].shift(L)
        usar = d.isna() & cand.notna()
        d[usar] = cand[usar]; k[usar] = L
    return d, k


s["d_i"], s["k_i"] = diferencia("log_i")
s["d_s"], s["k_s"] = diferencia("log_s")


def variables_mes(k):
    """Variables estacionales 'con huecos' de la autora: +1 en el mes, y en el mes anterior
    -1 (si k=1) o -1-k (si se saltó meses). Sin k el valor queda vacío, como en Stata."""
    X = {}
    for m in range(1, 13):
        prev = 12 if m == 1 else m - 1
        v = np.where(s.month == m, 1.0, 0.0)
        en_prev = s.month == prev
        v = np.where(en_prev & (k == 1), -1.0, v)
        v = np.where(en_prev & (k > 1), -1.0 - k, v)
        v = np.where(en_prev & k.isna(), np.nan, v)
        X[m] = v
    return pd.DataFrame(X, index=s.index)


def mco_ponderado(y, X, w):
    """Mínimos cuadrados ponderados (equivale a reg ... [pweight=...] de Stata)."""
    m = y.notna() & X.notna().all(axis=1) & w.notna()
    yy, XX, ww = y[m].to_numpy(), X[m].to_numpy(), w[m].to_numpy()
    sw = np.sqrt(ww)
    b, *_ = np.linalg.lstsq(XX * sw[:, None], yy * sw, rcond=None)
    res = yy - XX @ b
    ybar = np.average(yy, weights=ww)
    r2 = 1 - np.sum(ww * res ** 2) / np.sum(ww * (yy - ybar) ** 2)
    n, p = len(yy), XX.shape[1]
    return b, 1 - (1 - r2) * (n - 1) / (n - p), n


def factores(y, k):
    """Factor estacional por mes = coeficiente del mes - promedio de los 12 (enero = constante)."""
    M = variables_mes(k)
    X = pd.concat([pd.Series(1.0, index=s.index, name="cons"), M[list(range(2, 13))]], axis=1)
    b, r2a, n = mco_ponderado(y, X, s.pweight)
    coef = np.r_[b[0], b[1:]]                      # enero = constante; feb..dic = sus coeficientes
    f = coef - coef.mean()
    return f * 100, r2a, n


def trigonometrico(y):
    """Modelo alternativo: cambio mensual explicado por cos y sin del mes."""
    cos_ = np.cos(s.month * np.pi / 6); sin_ = np.sin(s.month * np.pi / 6)
    prev = np.where(s.month == 1, 12, s.month - 1)
    X = pd.DataFrame({"cons": 1.0, "d_cos": cos_ - np.cos(prev * np.pi / 6), "d_sin": sin_ - np.sin(prev * np.pi / 6)},
                     index=s.index)
    return mco_ponderado(y, X, s.pweight)[1]


fi, r2_i, n_i = factores(s.d_i, s.k_i)
fs, r2_s, n_s = factores(s.d_s, s.k_s)
ai, *_ = factores(s.iCoNA_solutionallHH.astype(float), s.k_i)     # factibilidad (en puntos %)
as_, *_ = factores(s.CoNA_solution.astype(float), s.k_s)


def prob_disponible(sol):
    """Ecuación 9: factibilidad sobre mes (enero base) y número de mercado, como en su código."""
    X = pd.concat([pd.Series(1.0, index=s.index, name="cons"),
                   pd.get_dummies(s.month, prefix="m", dtype=float).drop(columns="m_1"),
                   s.market_no.rename("mercado")], axis=1)
    b, *_ = mco_ponderado(sol.astype(float), X, s.pweight)
    return (b[0] + np.r_[0, b[1:12]]) * 100       # constante + efecto de cada mes (sin el término de mercado)


pi_, ps_ = prob_disponible(s.iCoNA_solutionallHH), prob_disponible(s.CoNA_solution)

tabla = pd.DataFrame({"Mes": MESES,
                      "Factor costo individual (%)": fi, "Factor costo compartido (%)": fs,
                      "Prob. dieta individual (%)": pi_, "Prob. dieta compartida (%)": ps_}).round(2)
brechas = {"Brecha estacional costo individual (pp)": fi.max() - fi.min(),
           "Brecha estacional costo compartido (pp)": fs.max() - fs.min(),
           "Brecha prob. dieta individual (pp)": pi_.max() - pi_.min(),
           "Brecha prob. dieta compartida (pp)": ps_.max() - ps_.min(),
           "R2 ajustado variables de mes, individual": r2_i, "R2 ajustado trigonométrico, individual": trigonometrico(s.d_i),
           "R2 ajustado variables de mes, compartido": r2_s, "R2 ajustado trigonométrico, compartido": trigonometrico(s.d_s)}
tabla.to_csv("resultados/tabla2_estacionalidad.csv", index=False)
pd.Series(brechas).round(4).to_csv("resultados/brechas_estacionales.csv", header=["valor"])
print(tabla.to_string(index=False)); print(pd.Series(brechas).round(3).to_string())
print(f"Observaciones: individual {n_i}, compartido {n_s}")

# ------------------------------------------------ Figura 4
AZUL_OSC, AZUL_CLA = "#1f3b73", "#6aa9e0"
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
for ax, prob, fac, tit in [(ejes[0], pi_, fi, "Dietas individualizadas"), (ejes[1], ps_, fs, "Olla compartida (hogar)")]:
    ax.bar(range(1, 13), prob, color=AZUL_CLA, label="% con dieta factible")
    ax.set_ylim(40, 100); ax.set_xticks(range(1, 13), MESES, fontsize=8); ax.set_title(tit)
    ax.set_ylabel("% de hogares con dieta factible")
    ax2 = ax.twinx(); ax2.plot(range(1, 13), fac, color=AZUL_OSC, marker="o", lw=2, label="Factor de costo")
    ax2.axhline(0, color="#999999", lw=0.8); ax2.set_ylim(-8, 8); ax2.set_ylabel("Cambio del costo vs. promedio (%)")
    ax.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
fig.suptitle("Figura 4 (réplica). Estacionalidad del costo y la factibilidad, 2013-2017")
fig.tight_layout(); fig.savefig("resultados/figura4_estacionalidad.png", dpi=200); plt.close(fig)
