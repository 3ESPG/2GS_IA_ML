"""
HeatSafe · Previsão de dia perigoso (uso em produção).

Carrega o modelo treinado no notebook e responde no formato do contrato da API:
    GET /previsao/amanha?estacao=A701  ->  {"data": "...", "perigoso": true, "probabilidade": 0.87}

Uso:
    python prever.py                       # prevê o dia seguinte ao último dia da base
    python prever.py --data 2023-11-12     # prevê 13/11/2023 com o que se sabia às 19h de 12/11
    python prever.py --horario outro.csv   # usa outro CSV horário no mesmo formato do INMET limpo

No serviço FastAPI (disciplina de SOA), basta importar `prever_amanha` e chamar com
as observações horárias do banco convertidas em DataFrame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

import atributos as at

PASTA = Path(__file__).parent
METADADOS = json.loads((PASTA / "modelo" / "metadados.json").read_text(encoding="utf-8"))
MODELO = joblib.load(PASTA / "modelo" / METADADOS["arquivo_modelo"])


def prever_amanha(horario: pd.DataFrame, data_d: str | pd.Timestamp | None = None) -> dict:
    """Usa as medições de 00h-18h do dia D (e dos dias anteriores) para prever o dia D+1."""
    x = at.montar_atributos(horario)
    data_d = x.index[-1] if data_d is None else pd.Timestamp(data_d).normalize()
    if data_d not in x.index:
        raise ValueError(f"Sem medições para {data_d:%d/%m/%Y} no CSV informado.")
    linha = x.loc[[data_d], METADADOS["atributos"]]
    if linha.isna().all(axis=1).iloc[0]:
        raise ValueError(f"O dia {data_d:%d/%m/%Y} não tem horas válidas suficientes até as {at.HORA_CORTE}h.")
    prob = float(MODELO.predict_proba(linha)[:, 1][0])
    return {
        "data": (data_d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "perigoso": prob >= METADADOS["limiar_decisao_probabilidade"],
        "probabilidade": round(prob, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="HeatSafe: amanhã será um dia de calor perigoso?")
    p.add_argument("--horario", default=PASTA / "dados" / "inmet_a701_horario_2015_2025.csv")
    p.add_argument("--data", help="dia D (AAAA-MM-DD); a previsão é para D+1. Padrão: último dia do CSV")
    args = p.parse_args()
    horario = pd.read_csv(args.horario, parse_dates=["data_hora_local"])
    print(json.dumps(prever_amanha(horario, args.data), ensure_ascii=False))


if __name__ == "__main__":
    main()
