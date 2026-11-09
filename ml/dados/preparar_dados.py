"""
Prepara os dados do INMET (estação A701 - São Paulo/Mirante) para o HeatSafe.

Entrada : brutos/*.CSV  (arquivos originais do INMET, um por ano)
Saída   : inmet_a701_horario_2015_2025.csv  (horário, limpo, com índice de calor)
          inmet_a701_diario_2015_2025.csv   (agregado por dia, horário de Brasília)

Uso: python preparar_dados.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

PASTA = Path(__file__).parent
BRUTOS = PASTA / "brutos"
MIN_HORAS_VALIDAS = 18  # dia com menos horas válidas de temperatura fica sem agregado

# As colunas mudam de nome entre os anos, mas a ordem é a mesma.
COLUNAS = [
    "data", "hora_utc", "precipitacao_mm", "pressao_mb", "pressao_max_mb", "pressao_min_mb",
    "radiacao_kj_m2", "temperatura_c", "ponto_orvalho_c", "temperatura_max_c", "temperatura_min_c",
    "orvalho_max_c", "orvalho_min_c", "umidade_max_pct", "umidade_min_pct", "umidade_pct",
    "vento_direcao_graus", "vento_rajada_ms", "vento_velocidade_ms",
]


def ler_ano(arquivo: Path) -> pd.DataFrame:
    df = pd.read_csv(arquivo, sep=";", skiprows=8, encoding="latin-1", decimal=",",
                     header=0, usecols=range(19), dtype=str)
    df.columns = COLUNAS
    # 2015-2018: "2015-01-01" e "00:00" | 2019+: "2019/01/01" e "0000 UTC"
    data = df["data"].str.replace("/", "-", regex=False)
    hora = df["hora_utc"].str.replace(" UTC", "", regex=False).str.replace(":", "", regex=False).str.zfill(4)
    df["data_hora_utc"] = pd.to_datetime(data + " " + hora, format="%Y-%m-%d %H%M", utc=True)
    numericas = COLUNAS[2:]
    df[numericas] = df[numericas].apply(lambda s: pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce"))
    df[numericas] = df[numericas].where(df[numericas] != -9999)
    return df.drop(columns=["data", "hora_utc"])


def indice_de_calor(temp_c: pd.Series, umidade: pd.Series) -> pd.Series:
    """Índice de calor da NOAA (Rothfusz + ajustes), entrada e saída em °C."""
    t = temp_c * 9 / 5 + 32
    rh = umidade
    simples = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    hi = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh
          - 0.00683783 * t ** 2 - 0.05481717 * rh ** 2 + 0.00122874 * t ** 2 * rh
          + 0.00085282 * t * rh ** 2 - 0.00000199 * t ** 2 * rh ** 2)
    ajuste_seco = ((13 - rh) / 4) * np.sqrt(np.clip((17 - np.abs(t - 95)) / 17, 0, None))
    hi = np.where((rh < 13) & (t >= 80) & (t <= 112), hi - ajuste_seco, hi)
    hi = np.where((rh > 85) & (t >= 80) & (t <= 87), hi + ((rh - 85) / 10) * ((87 - t) / 5), hi)
    hi = np.where((simples + t) / 2 >= 80, hi, simples)
    return pd.Series((hi - 32) * 5 / 9, index=temp_c.index).round(2)


def nivel(ic: pd.Series) -> pd.Series:
    return pd.cut(ic, bins=[-np.inf, 27, 32, 41, 54, np.inf], right=False,
                  labels=["NORMAL", "ATENCAO", "CAUTELA_EXTREMA", "PERIGO", "PERIGO_EXTREMO"])


def main() -> None:
    horario = pd.concat([ler_ano(f) for f in sorted(BRUTOS.glob("*.CSV"))], ignore_index=True)
    horario = horario.drop_duplicates("data_hora_utc").sort_values("data_hora_utc")
    horario["data_hora_local"] = horario["data_hora_utc"].dt.tz_convert("America/Sao_Paulo")
    # a conversão UTC -> Brasília puxa as últimas horas de 31/12/2014 do arquivo de 2015
    horario = horario[horario["data_hora_local"].dt.year.between(2015, 2025)].copy()
    horario["indice_calor_c"] = indice_de_calor(horario["temperatura_c"], horario["umidade_pct"])
    horario["nivel_risco"] = nivel(horario["indice_calor_c"])

    saida_h = horario[["data_hora_utc", "data_hora_local"] + COLUNAS[2:] + ["indice_calor_c", "nivel_risco"]].copy()
    saida_h["data_hora_utc"] = saida_h["data_hora_utc"].dt.strftime("%Y-%m-%d %H:%M")
    saida_h["data_hora_local"] = saida_h["data_hora_local"].dt.strftime("%Y-%m-%d %H:%M")
    saida_h.to_csv(PASTA / "inmet_a701_horario_2015_2025.csv", index=False)

    horario["data"] = horario["data_hora_local"].dt.date
    diario = horario.groupby("data").agg(
        horas_validas=("temperatura_c", "count"),
        temp_max_c=("temperatura_max_c", "max"),
        temp_min_c=("temperatura_min_c", "min"),
        temp_media_c=("temperatura_c", "mean"),
        umidade_media_pct=("umidade_pct", "mean"),
        umidade_min_pct=("umidade_min_pct", "min"),
        precipitacao_mm=("precipitacao_mm", "sum"),
        pressao_media_mb=("pressao_mb", "mean"),
        radiacao_total_kj_m2=("radiacao_kj_m2", "sum"),
        vento_medio_ms=("vento_velocidade_ms", "mean"),
        ponto_orvalho_medio_c=("ponto_orvalho_c", "mean"),
        indice_calor_max_c=("indice_calor_c", "max"),
    )
    incompletos = diario["horas_validas"] < MIN_HORAS_VALIDAS
    diario.loc[incompletos, diario.columns.drop("horas_validas")] = np.nan
    diario = diario.round(2)
    diario["nivel_risco_max"] = nivel(diario["indice_calor_max_c"])
    diario["dia_perigoso"] = (diario["indice_calor_max_c"] >= 32).astype("Int64").where(diario["indice_calor_max_c"].notna())
    diario.to_csv(PASTA / "inmet_a701_diario_2015_2025.csv")

    print(f"Horário: {len(saida_h)} linhas | {saida_h['data_hora_local'].min()} a {saida_h['data_hora_local'].max()}")
    print(f"Diário : {len(diario)} dias | {incompletos.sum()} dias incompletos (< {MIN_HORAS_VALIDAS}h válidas)")
    print(f"Faltantes temperatura: {horario['temperatura_c'].isna().mean():.1%} | umidade: {horario['umidade_pct'].isna().mean():.1%}")
    print(f"Dias perigosos (IC máx >= 32 °C): {int(diario['dia_perigoso'].sum())} de {int(diario['dia_perigoso'].notna().sum())} dias válidos")
    print(diario.groupby(pd.to_datetime(diario.index).year)["dia_perigoso"].agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
