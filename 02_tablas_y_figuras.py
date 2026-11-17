# -*- coding: utf-8 -*-
"""
PASO 2. Reproducir las tablas y figuras principales del artículo.

Sigue paso a paso la lógica del código Stata de la autora
("Schneider et al 2022_AJAE_Replication Code.do"), pero en Python.

Reglas que usa el artículo y que respetamos aquí:
  - Solo hogares rurales (reside == 1), rondas 2013 y 2016/17 (data_round 2 y 3)
    y con un mercado asignado (market_no no vacío).
  - Todas las medias y medianas usan el peso muestral del hogar (pweight),
    para que los resultados representen a toda la población rural de Malawi.
  - Los costos se pasan de kwachas a dólares PPP de 2011 dividiendo entre el
    factor PPP, para poder compararlos con la línea de pobreza internacional.

Salidas en resultados/: tabla1.csv, tabla2.csv, tabla3.csv, comparacion_con_paper.csv,
figura2_factibilidad.png, figura3_costo.png, figura5_acceso.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "resultados/"
LLAVES = ["case_id", "HHID", "y2_hhid", "y3_hhid", "data_round"]


# ------------------------------------------------ funciones con pesos
def media_p(x, w):
    """Media ponderada (equivale a 'svy: mean' de Stata)."""
    m = x.notna() & w.notna()
    return np.average(x[m], weights=w[m])


def mediana_p(x, w, q=0.5):
    """Mediana ponderada (equivale a 'epctile, svy p(50)' de Stata)."""
    m = x.notna() & w.notna()
    x, w = x[m].to_numpy(float), w[m].to_numpy(float)
    orden = np.argsort(x)
    x, w = x[orden], w[orden]
    acumulado = np.cumsum(w) / w.sum()
    return x[np.searchsorted(acumulado, q)]


def por_ronda(df, funcion, columna):
    """Calcula un indicador para 2013, 2016/17 y el total, como en las tablas del paper."""
    return [funcion(d[columna], d.pweight) for d in
            (df[df.data_round == 2], df[df.data_round == 3], df)]


# ------------------------------------------------ datos
hog = pd.read_csv("datos/hogares.csv.gz")
serie = pd.read_csv("datos/serie_mensual.csv.gz", low_memory=False)
comp = pd.read_csv("datos/composicion_hogar.csv.gz")
ppp = pd.read_csv("datos/ppp_mensual.csv.gz")
ppp_anual = ppp.groupby("year", as_index=False).PPP.first()   # factor PPP por año

# ================================================= TABLA 1: quiénes son los hogares
# Cuatro distritos estaban mal clasificados como rurales (en realidad son ciudad)
t1 = hog.copy()
t1.loc[(t1.reside == 1) & t1.district.isin([107, 210, 314, 315]), "reside"] = 0
t1 = t1[(t1.reside == 1) & (t1.data_round != 1) & t1.market_no.notna()]
t1 = t1.merge(comp, on=LLAVES, how="left").merge(ppp_anual, on="year", how="left")
t1["gasto_comida_pdpc"] = t1.totalexp_food / t1.PPP / 365 / t1.hhsize     # US$ PPP por persona al día
t1["gasto_total_pdpc"] = t1.totalexpenditure / t1.PPP / 365 / t1.hhsize
# Adultos y niños: la autora los promedia sobre las PERSONAS (cada persona lleva los totales
# de su hogar), no sobre los hogares. Por eso salen más altos que el promedio por hogar.
per = pd.read_csv("datos/personas.csv.gz").merge(comp, on=LLAVES, how="left")
per = per.merge(t1[LLAVES + ["pweight"]], on=LLAVES, how="inner")
tabla1 = pd.DataFrame({
    "Tamaño del hogar (media)": por_ronda(t1, media_p, "hhsize"),
    "Adultos >18 (media por persona)": por_ronda(per, media_p, "adulto"),
    "Niños <=18 (media por persona)": por_ronda(per, media_p, "nino"),
    "Proporción del gasto que va a comida (mediana)": por_ronda(t1, mediana_p, "pctfoodexp"),
    "Gasto en comida por persona al día, US$ PPP (mediana)": por_ronda(t1, mediana_p, "gasto_comida_pdpc"),
    "Gasto total por persona al día, US$ PPP (mediana)": por_ronda(t1, mediana_p, "gasto_total_pdpc"),
    "Número de hogares": [(t1.data_round == 2).sum(), (t1.data_round == 3).sum(), len(t1)],
    "Número de personas (sin bebés <6 meses)": [((per.data_round == r) & (per.age_sex_grp != 1) & (per.daysate_conv != 0)).sum()
                                               for r in (2, 3)] + [((per.age_sex_grp != 1) & (per.daysate_conv != 0)).sum()],
}, index=["2013", "2016/17", "Total"]).T.round(2)
tabla1.to_csv(R + "tabla1.csv")
print("\nTABLA 1. Características de los hogares rurales\n", tabla1)

# ================================================= TABLA 2: factibilidad y costo en toda la serie
s = serie[(serie.reside == 1) & (serie.year >= 2013)].copy()
s["indiv_pc"] = s.iCoNA_HH_pd_ppp / s.hhsize        # costo por persona al día (individualizado)
s["comp_pc"] = s.CoNA_sharing_pd_ppp / s.hhsize     # costo por persona al día (compartido)
s2 = s[(s.data_round != 1) & s.market_no.notna()]
tabla2 = pd.DataFrame({
    "Individualizado": [media_p(s2.iCoNA_solutionallHH * 100, s2.pweight), mediana_p(s2.indiv_pc, s2.pweight)],
    "Compartido (hogar)": [media_p(s2.CoNA_solution * 100, s2.pweight), mediana_p(s2.comp_pc, s2.pweight)],
}, index=["% de meses con dieta factible", "Costo mediano por persona al día (US$ PPP 2011)"]).round(3)
tabla2.to_csv(R + "tabla2.csv")
print(f"\nTABLA 2. Enero 2013 - julio 2017, {s2.hhxround.nunique()} hogares, {len(s2)} hogar-mes\n", tabla2)

# ================================================= TABLA 3: ¿pueden pagarla? (mes de la encuesta)
# Se toma el costo de la dieta en el mismo mes en que el hogar respondió la encuesta
c = s.copy()
c.loc[c.iCoNA_solutionallHH == 0, "iCoNA_HH_pd"] = np.nan
c = c[c.dateMY == c.date_consump_MY]
c = c[LLAVES + ["CoNA_solution", "iCoNA_solutionallHH", "CoNA_sharing_pd", "iCoNA_HH_pd",
                "CoNA_sharing_pd_ppp", "iCoNA_HH_pd_ppp"]]
h = hog[(hog.data_round != 1) & (hog.reside == 1)].drop(columns=["CoNA_sharing", "CoNA_sharing_pd"])
t3 = h.merge(c, on=LLAVES, how="inner")
t3 = t3[t3.market_no.notna()]

# Razón costo de la dieta / lo que el hogar gasta hoy. Mayor que 1 = no le alcanza.
t3["indiv_vs_comida"] = t3.iCoNA_HH_pd / (t3.totalexp_food / 365)
t3["indiv_vs_total"] = t3.iCoNA_HH_pd / (t3.totalexpenditure / 365)
t3["comp_vs_comida"] = t3.CoNA_sharing_pd / (t3.totalexp_food / 365)
t3["comp_vs_total"] = t3.CoNA_sharing_pd / (t3.totalexpenditure / 365)
for col in ["indiv_vs_comida", "indiv_vs_total", "comp_vs_comida", "comp_vs_total"]:
    t3["no_alcanza_" + col] = np.where(t3[col].isna(), np.nan, np.where(t3[col] > 1, 100, 0))
# Para la narrativa del paper: si no hay dieta factible ese mes, el hogar tampoco tiene acceso
for col, sol in [("indiv_vs_comida", "iCoNA_solutionallHH"), ("indiv_vs_total", "iCoNA_solutionallHH"),
                 ("comp_vs_comida", "CoNA_solution"), ("comp_vs_total", "CoNA_solution")]:
    t3["sin_acceso_" + col] = t3["no_alcanza_" + col]
    t3.loc[t3[sol] == 0, "sin_acceso_" + col] = 100
# Cuánto más cuesta compartir la olla frente a dietas individuales
t3["prima_compartir"] = t3.CoNA_sharing_pd / t3.iCoNA_HH_pd - 1

tabla3 = pd.DataFrame({
    "Individualizado": [media_p(t3.iCoNA_solutionallHH * 100, t3.pweight),
                        mediana_p(t3.iCoNA_HH_pd_ppp, t3.pweight),
                        mediana_p(t3.indiv_vs_comida, t3.pweight), mediana_p(t3.indiv_vs_total, t3.pweight),
                        media_p(t3.sin_acceso_indiv_vs_comida, t3.pweight),
                        media_p(t3.sin_acceso_indiv_vs_total, t3.pweight)],
    "Compartido (hogar)": [media_p(t3.CoNA_solution * 100, t3.pweight),
                           mediana_p(t3.CoNA_sharing_pd_ppp, t3.pweight),
                           mediana_p(t3.comp_vs_comida, t3.pweight), mediana_p(t3.comp_vs_total, t3.pweight),
                           media_p(t3.sin_acceso_comp_vs_comida, t3.pweight),
                           media_p(t3.sin_acceso_comp_vs_total, t3.pweight)],
}, index=["% hogares con dieta factible en el mes de la encuesta",
          "Costo mediano del hogar al día (US$ PPP 2011)",
          "Costo / gasto actual en comida (mediana)",
          "Costo / gasto total (mediana)",
          "% sin acceso sin gastar más en comida",
          "% sin acceso aunque gaste todo su presupuesto"]).round(3)
tabla3.loc["Prima por compartir la olla (mediana, %)"] = [np.nan, round(mediana_p(t3.prima_compartir * 100, t3.pweight), 2)]
tabla3.to_csv(R + "tabla3.csv")
print(f"\nTABLA 3. Asequibilidad en el mes de la encuesta ({len(t3)} hogares)\n", tabla3)

# ================================================= FIGURAS 2 y 3: mes a mes
# Su código filtra con "dateMY<2017-07", pero en Stata eso es el número 2010 y no una fecha,
# así que no filtra nada: la figura del paper incluye julio de 2017. Hacemos lo mismo.
fig_df = s.copy()
fig_df.loc[fig_df.CoNA_sharing.isna(), "CoNA_solution"] = 0
mensual = fig_df.groupby("dateMY").apply(lambda d: pd.Series({
    "factible_indiv": media_p(d.iCoNA_solutionallHH * 100, d.pweight),
    "factible_comp": media_p(d.CoNA_solution * 100, d.pweight),
    "costo_indiv": mediana_p(d.indiv_pc, d.pweight),
    "costo_comp": mediana_p(d.comp_pc, d.pweight)}), include_groups=False).reset_index()
mensual["dateMY"] = pd.to_datetime(mensual.dateMY)
mensual.to_csv(R + "serie_mensual_resumen.csv", index=False)

AZUL_OSC, AZUL_CLA = "#1f3b73", "#6aa9e0"
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(mensual.dateMY, mensual.factible_indiv, width=25, color=AZUL_OSC, label="Dietas individualizadas")
ax.bar(mensual.dateMY, mensual.factible_comp, width=25, color=AZUL_CLA, label="Olla compartida (hogar)")
# Líneas de referencia del paper: factibilidad media de cada escenario (Tabla 2)
ax.axhline(tabla2.iloc[0, 0], color="#333333", ls="--", lw=1)
ax.axhline(tabla2.iloc[0, 1], color="#333333", ls="-", lw=1)
ax.set_ylabel("% de hogares con dieta factible"); ax.set_ylim(0, 100)
ax.set_title("Figura 2 (réplica). Hogares rurales con una dieta nutricionalmente adecuada disponible, 2013-2017")
ax.legend(loc="lower left", frameon=False); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(R + "figura2_factibilidad.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(mensual.dateMY, mensual.costo_indiv, color=AZUL_OSC, lw=2, label="Dietas individualizadas")
ax.plot(mensual.dateMY, mensual.costo_comp, color=AZUL_CLA, lw=2, ls="--", label="Olla compartida (hogar)")
# Líneas de referencia del paper: costo mediano de cada escenario (Tabla 2)
ax.axhline(tabla2.iloc[1, 0], color=AZUL_OSC, ls="-", lw=1)
ax.axhline(tabla2.iloc[1, 1], color=AZUL_CLA, ls="--", lw=1)
ax.set_ylabel("Costo mediano por persona al día\n(US$ PPP 2011)")
ax.set_title("Figura 3 (réplica). Costo de la dieta adecuada mes a mes, 2013-2017")
ax.legend(loc="upper left", frameon=False); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(R + "figura3_costo.png", dpi=200); plt.close(fig)

# ================================================= FIGURA 5: acceso (barras)
vals = np.array([[media_p(t3.sin_acceso_indiv_vs_comida, t3.pweight), media_p(t3.sin_acceso_indiv_vs_total, t3.pweight)],
                 [media_p(t3.sin_acceso_comp_vs_comida, t3.pweight), media_p(t3.sin_acceso_comp_vs_total, t3.pweight)]])
fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = np.arange(2)
for j, (lab, col) in enumerate([("Sin gastar más en comida", AZUL_OSC), ("Aunque gaste todo su presupuesto", AZUL_CLA)]):
    b = ax.bar(x + (j - 0.5) * 0.38, vals[:, j], 0.38, color=col, label=lab)
    ax.bar_label(b, fmt="%.0f%%", padding=2)
ax.set_xticks(x, ["Dietas individualizadas", "Olla compartida (hogar)"])
ax.set_ylabel("% de hogares sin acceso a una dieta adecuada"); ax.set_ylim(0, 100)
ax.set_title("Figura 5 (réplica). Hogares sin acceso a una dieta adecuada")
ax.legend(frameon=False, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(R + "figura5_acceso.png", dpi=200); plt.close(fig)
print("\nFigura 5, % sin acceso [sin gastar más en comida, aunque gaste todo]:\n", vals.round(1))
