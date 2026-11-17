# -*- coding: utf-8 -*-
"""
PASO 4. Dietas individualizadas (cota inferior del artículo) con nuestro propio solver.

En este escenario cada persona come su propia dieta ideal. La autora resolvió
una programación lineal por cada grupo de edad y sexo (y nivel de actividad),
en cada mercado y cada mes (archivo "8 - Individual CoNAs" de su GitHub).
Luego sumó, dentro de cada hogar, el costo de cada miembro (archivo "9r - Merge data").

Aquí hacemos lo mismo en Python:
  1) Resolvemos el PL de cada grupo x mercado x mes desde 2013.
  2) Lo comparamos con el costo individual que publicó la autora.
  3) Armamos el costo del hogar: suma de sus miembros, ajustada por los días que
     comen en casa. Si a un solo miembro no le sale dieta, el hogar no tiene dieta.
  4) Lo comparamos con el iCoNA_HH_pd de la autora.

Requerimientos individuales: datos/requerimientos_individuales.csv.gz, sacados del
archivo HHNeeds_IndivLevel_r.dta de Dataverse (primer valor por grupo, como su collapse).
"""
import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog

NUTRIENTES = ["energy", "carbohydrate", "protein", "lipids", "vitA", "retinol", "vitC", "vitE",
              "thiamin", "riboflavin", "niacin", "vitB6", "folate", "vitB12", "calcium", "copper",
              "iron", "magnesium", "phosphorus", "selenium", "zinc", "sodium"]  # nutr_no 2..23
LLAVES = ["case_id", "HHID", "y2_hhid", "y3_hhid", "data_round"]

req = pd.read_csv("datos/requerimientos_individuales.csv.gz")
precios = pd.read_csv("datos/precios_mercados.csv.gz")

# Igual que la autora: EAR vacío para carbohidratos, grasas y sodio; UL y AMDR en 0 = no aplica
req.loc[(req.ear_perweek == 0) & req.nutr_no.isin([5, 7, 23]), "ear_perweek"] = np.nan
for c in ["ul_perweek", "amdr_lower_perweek", "amdr_upper_perweek"]:
    req.loc[req[c] == 0, c] = np.nan

grupos = {}
for (g, pal), d in req.groupby(["age_sex_grp", "pal"]):
    d = d.set_index("nutr_no")
    vec = lambda col: np.array([d[col].get(k, np.nan) for k in range(2, 24)], float)
    grupos[(g, pal)] = (vec("ear_perweek"), vec("ul_perweek"), vec("amdr_lower_perweek"), vec("amdr_upper_perweek"))

matrices = {}
for clave, d in precios.groupby(["market_no", "year", "month"]):
    d = d[d.uni_price_net.notna() & (d.uni_price_net > 0)]
    matrices[clave] = (d.uni_price_net.values / 1000.0, d[NUTRIENTES].fillna(0).values.T / 100.0)


def resolver(costo, A, EAR, UL, LOW, UP):
    filas, lim = [], []
    for k in range(1, 22):
        for valor, signo in ((EAR[k], -1), (LOW[k], -1), (UL[k], 1), (UP[k], 1)):
            if not np.isnan(valor):
                filas.append(signo * A[k]); lim.append(signo * valor)
    r = linprog(costo, A_ub=np.array(filas), b_ub=np.array(lim), A_eq=A[:1], b_eq=[EAR[0]],
                bounds=(0, None), method="highs")
    return r.fun if r.status == 0 else np.nan


# ------------------------------------------------ 1) un PL por grupo, mercado y mes
inicio = time.time()
filas = []
for (m, a, mes), (c, A) in matrices.items():
    for (g, pal), reqs in grupos.items():
        filas.append((g, pal, m, a, mes, resolver(c, A, *reqs)))
res = pd.DataFrame(filas, columns=["age_sex_grp", "pal", "market_no", "year", "month", "iCoNA_replica"])
# Hallazgo: en el script de R de la autora ("8 - Individual CoNAs_R.R") el bucle de los bebés de
# 6 a 12 meses (grupo 2) va de 1 a 3048 y no hasta 3683, así que los mercados 25 a 29 quedaron
# sin resolver y figuran como "sin dieta". Para reproducir sus cifras hacemos lo mismo.
res["iCoNA_sin_ajuste"] = res.iCoNA_replica
res.loc[(res.age_sex_grp == 2) & (res.market_no >= 25), "iCoNA_replica"] = np.nan
print(f"{len(res)} programas lineales individuales en {time.time() - inicio:.0f} s")

# ------------------------------------------------ 2) comparar con la autora
aut = pd.read_csv("datos/icona_autora.csv.gz")
comp = aut.merge(res, on=["age_sex_grp", "pal", "market_no", "year", "month"], how="inner")
fa, fr = comp.iCoNA.notna(), comp.iCoNA_replica.notna()
ambos = fa & fr
dif = comp.loc[ambos, "iCoNA_replica"] / comp.loc[ambos, "iCoNA"] - 1
print(f"Comparados: {len(comp)} | factibles autora {fa.mean():.2%}, réplica {fr.mean():.2%}, "
      f"coincide {(fa == fr).mean():.2%}")
print(f"Diferencia relativa de costo: mediana {dif.abs().median():.2e}, p99 {dif.abs().quantile(.99):.2e}, "
      f"corr {np.corrcoef(comp.loc[ambos, 'iCoNA_replica'], comp.loc[ambos, 'iCoNA'])[0, 1]:.6f}")

# ------------------------------------------------ 3) costo individual del hogar
per = pd.read_csv("datos/personas.csv.gz")
per = per[per.data_round != 1]
serie = pd.read_csv("datos/serie_mensual.csv.gz", low_memory=False, parse_dates=["dateMY"])
serie = serie[(serie.reside == 1) & (serie.data_round != 1) & serie.market_no.notna()]
res["data_round"] = np.where(res.year <= 2015, 2, 3)


def costo_hogar(tabla, col, nombre):
    """Suma el costo diario de los miembros; si a uno le falta dieta, el hogar queda sin dieta."""
    x = per.merge(tabla, on=["age_sex_grp", "market_no", "pal", "data_round"], how="inner")
    x["costo_dia"] = x[col] * x.daysate_conv / 7
    hh = x.groupby(LLAVES + ["year", "month"], dropna=False).agg(
        miembros=("PID", "size"), con_sol=("costo_dia", "count"), costo=("costo_dia", "sum")).reset_index()
    hh.loc[hh.miembros != hh.con_sol, "costo"] = np.nan
    return hh.rename(columns={"costo": nombre})[LLAVES + ["year", "month", nombre]]


# Variante de control: igual que la réplica, pero con el costo publicado por la autora para los
# niños de 1 a 3 años (grupo 3), que no se puede reproducir con sus requerimientos públicos.
res_ctrl = res.merge(aut[["age_sex_grp", "pal", "market_no", "year", "month", "iCoNA"]],
                     on=["age_sex_grp", "pal", "market_no", "year", "month"], how="left")
res_ctrl["costo_ctrl"] = np.where(res_ctrl.age_sex_grp == 3, res_ctrl.iCoNA, res_ctrl.iCoNA_replica)
serie["year"], serie["month"] = serie.dateMY.dt.year, serie.dateMY.dt.month
s = serie.merge(costo_hogar(res, "iCoNA_replica", "iCoNA_HH_pd_replica"), on=LLAVES + ["year", "month"], how="left")
s = s.merge(costo_hogar(res_ctrl, "costo_ctrl", "iCoNA_HH_pd_ctrl"), on=LLAVES + ["year", "month"], how="left")
fa, fr = s.iCoNA_HH_pd.notna(), s.iCoNA_HH_pd_replica.notna()
ambos = fa & fr
dif = s.loc[ambos, "iCoNA_HH_pd_replica"] / s.loc[ambos, "iCoNA_HH_pd"] - 1
print(f"Hogar-mes: {len(s)} | factibles autora {fa.mean():.2%}, réplica {fr.mean():.2%}, coincide {(fa == fr).mean():.2%}")
print(f"Costo del hogar: diferencia mediana {dif.abs().median():.2e}, corr "
      f"{np.corrcoef(s.loc[ambos, 'iCoNA_HH_pd_replica'], s.loc[ambos, 'iCoNA_HH_pd'])[0, 1]:.6f}")
res.to_csv("resultados/dietas_individuales_por_grupo.csv.gz", index=False)
s[["hhxround", "dateMY", "pweight", "hhsize", "PPP_month_interpolated", "iCoNA_HH_pd", "iCoNA_HH_pd_replica", "iCoNA_HH_pd_ctrl"]].to_csv(
    "resultados/dietas_individuales_hogar.csv.gz", index=False)

# ------------------------------------------------ 5) Tabla 2 con nuestras dietas individuales
def media_p(x, w):
    m = x.notna() & w.notna(); return np.average(x[m], weights=w[m])


def mediana_p(x, w):
    m = x.notna() & w.notna()
    x, w = x[m].to_numpy(float), w[m].to_numpy(float)
    o = np.argsort(x); x, w = x[o], w[o]
    return x[np.searchsorted(np.cumsum(w) / w.sum(), 0.5)]


s["pc_autora"] = s.iCoNA_HH_pd / s.PPP_month_interpolated / s.hhsize
s["pc_replica"] = s.iCoNA_HH_pd_replica / s.PPP_month_interpolated / s.hhsize
s["pc_ctrl"] = s.iCoNA_HH_pd_ctrl / s.PPP_month_interpolated / s.hhsize
t = pd.DataFrame({
    "Autora (sus costos)": [media_p(s.iCoNA_HH_pd.notna().astype(float) * 100, s.pweight), mediana_p(s.pc_autora, s.pweight)],
    "Réplica (nuestro solver)": [media_p(s.iCoNA_HH_pd_replica.notna().astype(float) * 100, s.pweight), mediana_p(s.pc_replica, s.pweight)],
    "Réplica con grupo 3 de la autora": [media_p(s.iCoNA_HH_pd_ctrl.notna().astype(float) * 100, s.pweight), mediana_p(s.pc_ctrl, s.pweight)],
}, index=["% de meses con dieta individual factible", "Costo mediano por persona al día (US$ PPP)"]).round(3)
t.to_csv("resultados/tabla2_individual_replica.csv")
print("\nTabla 2, columna individual:\n" + t.to_string())
