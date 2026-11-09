"""
HeatSafe · Atributos do modelo de previsão de dia perigoso.

Fonte única da engenharia de atributos: usada pelo notebook (treino) e pelo
`prever.py` / serviço de previsão (SOA) em produção. Assim, o modelo recebe
exatamente as mesmas colunas no treino e no uso real.

Regra de ouro: o modelo roda às 19h do dia D (véspera). Por isso, os atributos
do dia D usam só as medições de 00h às 18h (horário de Brasília), que é o que
de fato já existe às 19h. Nada do dia D+1 entra nos atributos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORA_CORTE = 18          # última hora do dia D usada (o modelo roda às 19h)
MIN_HORAS_JANELA = 14    # menos que isso (de 19 possíveis) = dia sem dado
LIMIAR_ALVO_C = 30.0     # amanhã é "dia perigoso" se o índice de calor máximo na estação for >= 30 °C

# Ordem fixa das colunas que o modelo espera (não mudar sem retreinar)
ATRIBUTOS = [
    # dia D (00h-18h)
    "ic_max", "temp_max", "temp_min", "temp_media", "temp_18h",
    "umid_media", "umid_min", "orvalho_medio", "pressao_media", "pressao_18h",
    "radiacao_total", "chuva", "vento_medio",
    # memória dos dias anteriores (D-1 e D-2)
    "ic_max_d1", "ic_max_d2", "temp_max_d1", "temp_max_d2",
    "umid_media_d1", "umid_media_d2", "orvalho_medio_d1", "orvalho_medio_d2",
    "pressao_media_d1", "pressao_media_d2",
    # tendência
    "ic_max_media3", "ic_max_media7", "delta_ic", "delta_pressao",
    # calendário (sazonalidade)
    "mes", "doy_sin", "doy_cos",
]


def indice_de_calor(temp_c, umidade_pct):
    """Índice de calor da NOAA (regressão de Rothfusz + ajustes). Entrada e saída em °C.

    É a mesma fórmula usada pelo dispositivo ESP32 do HeatSafe e por `dados/preparar_dados.py`.
    """
    temp_c = pd.Series(temp_c, dtype=float)
    rh = pd.Series(umidade_pct, dtype=float, index=temp_c.index)
    t = temp_c * 9 / 5 + 32
    simples = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    hi = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh
          - 0.00683783 * t ** 2 - 0.05481717 * rh ** 2 + 0.00122874 * t ** 2 * rh
          + 0.00085282 * t * rh ** 2 - 0.00000199 * t ** 2 * rh ** 2)
    ajuste_seco = ((13 - rh) / 4) * np.sqrt(np.clip((17 - np.abs(t - 95)) / 17, 0, None))
    hi = np.where((rh < 13) & (t >= 80) & (t <= 112), hi - ajuste_seco, hi)
    hi = np.where((rh > 85) & (t >= 80) & (t <= 87), hi + ((rh - 85) / 10) * ((87 - t) / 5), hi)
    hi = np.where((simples + t) / 2 >= 80, hi, simples)
    return pd.Series((hi - 32) * 5 / 9, index=temp_c.index).round(2)


def resumo_diario_ate_18h(horario: pd.DataFrame) -> pd.DataFrame:
    """Resume cada dia usando só as horas de 00h a HORA_CORTE (horário local).

    `horario` precisa ter as colunas do CSV horário do INMET já limpo
    (`dados/inmet_a701_horario_2015_2025.csv`).
    """
    h = horario.copy()
    h["data_hora_local"] = pd.to_datetime(h["data_hora_local"])
    if "indice_calor_c" not in h:
        h["indice_calor_c"] = indice_de_calor(h["temperatura_c"], h["umidade_pct"])
    h["data"] = h["data_hora_local"].dt.normalize()
    h["hora"] = h["data_hora_local"].dt.hour
    janela = h[h["hora"] <= HORA_CORTE]
    g = janela.groupby("data")

    dia = pd.DataFrame({
        "horas_validas": g["temperatura_c"].count(),
        "ic_max": g["indice_calor_c"].max(),
        "temp_max": g["temperatura_max_c"].max(),
        "temp_min": g["temperatura_min_c"].min(),
        "temp_media": g["temperatura_c"].mean(),
        "umid_media": g["umidade_pct"].mean(),
        "umid_min": g["umidade_min_pct"].min(),
        "orvalho_medio": g["ponto_orvalho_c"].mean(),
        "pressao_media": g["pressao_mb"].mean(),
        "radiacao_total": g["radiacao_kj_m2"].sum(min_count=1),
        "chuva": g["precipitacao_mm"].sum(min_count=1),
        "vento_medio": g["vento_velocidade_ms"].mean(),
    })
    as_18h = janela[janela["hora"] == HORA_CORTE].set_index("data")
    dia["temp_18h"] = as_18h["temperatura_c"]
    dia["pressao_18h"] = as_18h["pressao_mb"]

    incompletos = dia["horas_validas"] < MIN_HORAS_JANELA
    dia.loc[incompletos, dia.columns.drop("horas_validas")] = np.nan
    # garante um registro por dia, mesmo nos dias sem nenhuma medição
    return dia.asfreq("D")


def montar_atributos(horario: pd.DataFrame) -> pd.DataFrame:
    """Gera a tabela de atributos (uma linha por dia D) na ordem de ATRIBUTOS.

    Cada linha descreve o que se sabe às 19h do dia D e serve para prever o dia D+1.
    Valores faltantes ficam como NaN (o pipeline do modelo preenche com a mediana do treino).
    """
    dia = resumo_diario_ate_18h(horario)
    x = dia.drop(columns="horas_validas").copy()
    for col in ["ic_max", "temp_max", "umid_media", "orvalho_medio", "pressao_media"]:
        for k in (1, 2):
            x[f"{col}_d{k}"] = dia[col].shift(k)
    x["ic_max_media3"] = dia["ic_max"].rolling(3, min_periods=2).mean()
    x["ic_max_media7"] = dia["ic_max"].rolling(7, min_periods=4).mean()
    x["delta_ic"] = dia["ic_max"] - dia["ic_max"].shift(1)
    x["delta_pressao"] = dia["pressao_media"] - dia["pressao_media"].shift(1)
    dia_do_ano = x.index.dayofyear
    x["mes"] = x.index.month
    x["doy_sin"] = np.sin(2 * np.pi * dia_do_ano / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * dia_do_ano / 365.25)
    x.index.name = "data"
    return x[ATRIBUTOS]


def montar_alvo(diario: pd.DataFrame, limiar: float = LIMIAR_ALVO_C) -> pd.Series:
    """Alvo da linha do dia D: 1 se o índice de calor máximo do dia D+1 (dia inteiro) >= limiar.

    `diario` é o CSV diário (`dados/inmet_a701_diario_2015_2025.csv`) indexado por data.
    Dias D+1 sem dado viram NaN e são descartados no treino.
    """
    ic_amanha = diario["indice_calor_max_c"].asfreq("D").shift(-1)
    return (ic_amanha >= limiar).astype("Int64").where(ic_amanha.notna()).rename("alvo")
