# -*- coding: utf-8 -*-
"""
PASO 0. Preparar los datos de la réplica.

Lee los archivos originales que la autora publicó en Harvard Dataverse
(https://doi.org/10.7910/DVN/AXDJZ8) y guarda solo las columnas que usamos,
en archivos pequeños .csv.gz dentro de la carpeta datos/. Así el repositorio
se puede correr sin descargar los 1,4 GB originales.

Uso:  python 00_preparar_datos.py  RUTA_A_LOS_ARCHIVOS_DE_DATAVERSE
Archivos que necesita en esa ruta:
  MalawiIHPS_DietQualCost_HH_r.dta               (hogares, gasto, mes de encuesta)
  MalawiIHPS_DietQualCost_CoNAiCoNAseries_r.dta  (costos de la autora mes a mes)
  MalawiIHPS_DietQualCost_PID_r.dta              (personas y requerimientos del hogar)
  CPIDataForCoNA.dta  o  CPIDataForCoNA_2013_2017.csv.gz  (precios por mercado y mes)
  PPP_Denton.dta                                 (conversión de kwachas a dólares PPP)
"""
import sys, os
import pandas as pd

ORIGEN = sys.argv[1] if len(sys.argv) > 1 else "datos_dataverse"
DESTINO = "datos"
os.makedirs(DESTINO, exist_ok=True)

# Nutrientes en el orden que usa la autora (nutr_no 2 a 23; el 1 es el precio)
NUTRIENTES = ["energy", "carbohydrate", "protein", "lipids", "vitA", "retinol", "vitC", "vitE",
              "thiamin", "riboflavin", "niacin", "vitB6", "folate", "vitB12", "calcium", "copper",
              "iron", "magnesium", "phosphorus", "selenium", "zinc", "sodium"]


def leer(nombre, columnas=None):
    return pd.read_stata(os.path.join(ORIGEN, nombre), columns=columnas, convert_categoricals=False)


# 1) Hogares: datos de la encuesta y gasto anual en comida y total
hogares = leer("MalawiIHPS_DietQualCost_HH_r.dta", [
    "case_id", "HHID", "y2_hhid", "y3_hhid", "data_round", "pweight", "ea_id", "district", "reside",
    "year", "hhsize", "market_no", "dateMY", "date_consump_MY", "totalexp_food", "totalexpenditure",
    "pctfoodexp", "CoNA_sharing", "CoNA_sharing_pd"])
hogares.to_csv(f"{DESTINO}/hogares.csv.gz", index=False)

# 2) Serie mensual 2013-2017 con los costos que calculó la autora
serie = leer("MalawiIHPS_DietQualCost_CoNAiCoNAseries_r.dta", [
    "case_id", "HHID", "y2_hhid", "y3_hhid", "hhxround", "data_round", "year", "month", "dateMY",
    "date_consump_MY", "market_no", "reside", "pweight", "ea_id", "hhsize", "KCAL_hh_perday",
    "CoNA_sharing", "CoNA_sharing_pd", "CoNA_sharing_pd_ppp", "CoNA_solution",
    "iCoNA_HH_pd", "iCoNA_HH_pd_ppp", "iCoNA_solutionallHH", "PPP", "PPP_month_interpolated"])
serie = serie[serie.year >= 2013]
serie.to_csv(f"{DESTINO}/serie_mensual.csv.gz", index=False)

# 3) Requerimientos compartidos de cada hogar (por semana), uno por hogar
cols = ["hhxround", "data_round", "market_no", "hhsize", "KCAL_hh_perweek"]
for k in range(2, 24):
    cols += [f"EAR_sharing_{k}", f"UL_sharing_{k}", f"AMDRlow_sharing_{k}", f"AMDRup_sharing_{k}"]
req = leer("MalawiIHPS_DietQualCost_PID_r.dta", cols).drop_duplicates("hhxround")
req.to_csv(f"{DESTINO}/requerimientos_hogar.csv.gz", index=False)

# 3b) Composición del hogar: cuántos adultos (>18) y niños (<=18) tiene (para la Tabla 1)
per = leer("MalawiIHPS_DietQualCost_PID_r.dta",
           ["case_id", "HHID", "y2_hhid", "y3_hhid", "data_round", "age_sex_grp"])
per["adulto"] = per.age_sex_grp.isin([10, 11, 12, 13, 16, 17, 18, 19, 24, 25]).astype(int)
per["nino"] = per.age_sex_grp.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 23]).astype(int)
llaves = ["case_id", "HHID", "y2_hhid", "y3_hhid", "data_round"]
comp = per.groupby(llaves, dropna=False)[["adulto", "nino"]].sum().reset_index()
comp.to_csv(f"{DESTINO}/composicion_hogar.csv.gz", index=False)

# 4) Precios por mercado y mes (desde 2013) con la composición de cada alimento
csv = os.path.join(ORIGEN, "CPIDataForCoNA_2013_2017.csv.gz")
if os.path.exists(csv):
    precios = pd.read_csv(csv)
else:
    precios = leer("CPIDataForCoNA.dta", ["market_no", "market", "year", "month", "food_no",
                                          "food_item", "uni_price_net"] + NUTRIENTES)
    precios = precios[precios.year >= 2013]
precios.to_csv(f"{DESTINO}/precios_mercados.csv.gz", index=False)

# 5) Factores PPP por mes
leer("PPP_Denton.dta").to_csv(f"{DESTINO}/ppp_mensual.csv.gz", index=False)

print("Listo: datos guardados en", DESTINO)
