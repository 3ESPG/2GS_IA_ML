# Dados climáticos · INMET, estação A701 (São Paulo – Mirante)

Dados **reais e públicos** usados no HeatSafe pela parte de Inteligência Artificial & Machine Learning (e pelo serviço de previsão da disciplina de SOA). **Nenhum dado aqui é simulado.**

## Fonte

| Item | Valor |
|---|---|
| Instituição | INMET (Instituto Nacional de Meteorologia) |
| Base | BDMEP: dados históricos anuais das estações automáticas |
| Link | https://portal.inmet.gov.br/dadoshistoricos |
| Arquivos baixados | `https://portal.inmet.gov.br/uploads/dadoshistoricos/<ANO>.zip`, de 2015 a 2025 |
| Data do download | 08/11/2026 |
| Estação | A701, SÃO PAULO - MIRANTE (estação automática, fundada em 25/07/2006) |
| Localização | Lat −23,4833 · Lon −46,6167 · Altitude 792 m · Zona Norte de SP (Santana) |
| Período | 01/01/2015 a 31/12/2025, dados **horários** |
| Licença | Dados públicos do governo federal (citar o INMET como fonte) |

Cada zip anual traz um CSV por estação do Brasil. Extraímos só o arquivo da A701.

## Arquivos desta pasta

| Arquivo | O que é |
|---|---|
| `brutos/` | Os 11 CSVs originais do INMET, sem nenhuma alteração (um por ano) |
| `inmet_a701_horario_2015_2025.csv` | Todos os anos juntos e limpos: **96.430 horas**, com índice de calor e nível de risco. **É a entrada do modelo** |
| `inmet_a701_diario_2015_2025.csv` | Agregado por dia inteiro (horário de Brasília): **4.018 dias**. Usado para o **alvo** (índice de calor máximo do dia seguinte) e para a análise exploratória |
| `preparar_dados.py` | Script que gera os dois CSVs acima a partir de `brutos/` |

Para gerar de novo: `pip install pandas numpy` e depois `python preparar_dados.py` (dentro desta pasta).

## Tratamento feito

1. **Leitura:** CSV com `;`, vírgula decimal e codificação latin-1. As 8 primeiras linhas (metadados da estação) são puladas.
2. **Nomes das colunas:** o INMET muda os nomes entre os anos (e o formato da data muda em 2019: `2015-01-01` / `00:00` vira `2019/01/01` / `0000 UTC`). As colunas foram renomeadas pela posição, que é a mesma em todos os anos.
3. **Faltantes:** o valor `-9999` foi trocado por vazio (NaN). Só 0,4% das horas não têm temperatura ou umidade.
4. **Fuso horário:** o INMET registra em UTC. Convertemos para o horário de Brasília (`America/Sao_Paulo`, já considerando o horário de verão que existia até 2019) para os dias começarem à meia-noite local.
5. **Índice de calor:** calculado hora a hora com a fórmula da NOAA (regressão de Rothfusz + ajustes), a mesma que o dispositivo ESP32 usa. O nível de risco segue a mesma tabela do HeatSafe.
6. **Agregação diária:** dias com menos de 18 horas válidas de temperatura ficaram com os valores vazios (22 dias no total).

## Colunas

### Horário (`inmet_a701_horario_2015_2025.csv`)

| Coluna | Descrição |
|---|---|
| `data_hora_utc`, `data_hora_local` | Data e hora da medição (UTC e Brasília) |
| `precipitacao_mm` | Chuva na hora (mm) |
| `pressao_mb`, `pressao_max_mb`, `pressao_min_mb` | Pressão atmosférica (mB) |
| `radiacao_kj_m2` | Radiação solar global (kJ/m²); vazia à noite |
| `temperatura_c`, `temperatura_max_c`, `temperatura_min_c` | Temperatura do ar: na hora, máxima e mínima da hora anterior (°C) |
| `ponto_orvalho_c`, `orvalho_max_c`, `orvalho_min_c` | Ponto de orvalho (°C) |
| `umidade_pct`, `umidade_max_pct`, `umidade_min_pct` | Umidade relativa (%) |
| `vento_direcao_graus`, `vento_rajada_ms`, `vento_velocidade_ms` | Vento |
| `indice_calor_c` | **Índice de calor** (sensação térmica, °C), calculado |
| `nivel_risco` | NORMAL, ATENCAO, CAUTELA_EXTREMA, PERIGO, PERIGO_EXTREMO, calculado |

### Diário (`inmet_a701_diario_2015_2025.csv`)

| Coluna | Descrição |
|---|---|
| `data` | Dia (horário de Brasília) |
| `horas_validas` | Quantas horas do dia têm temperatura |
| `temp_max_c`, `temp_min_c`, `temp_media_c` | Temperatura do dia (°C) |
| `umidade_media_pct`, `umidade_min_pct` | Umidade do dia (%) |
| `precipitacao_mm` | Chuva total do dia (mm) |
| `pressao_media_mb` | Pressão média (mB) |
| `radiacao_total_kj_m2` | Radiação solar total (kJ/m²) |
| `vento_medio_ms` | Velocidade média do vento (m/s) |
| `ponto_orvalho_medio_c` | Ponto de orvalho médio (°C) |
| `indice_calor_max_c` | **Maior índice de calor do dia** (°C) |
| `nivel_risco_max` | Nível de risco do pior momento do dia |
| `dia_perigoso` | 1 se `indice_calor_max_c` ≥ 32 °C (Cautela extrema, na escala do HeatSafe), 0 se não |

## Visão geral dos dados

- **Índice de calor máximo diário:** média de 25,7 °C, mínimo de 10,6 °C, máximo de **35,5 °C em 15/11/2023** (onda de calor de novembro de 2023).
- **Dias com índice ≥ 30 °C** (alvo do modelo): **669 de 3.996 dias válidos (16,7%)**.
- **Dias com índice ≥ 32 °C:** 169 de 3.996 (4,2%).
- **O calor perigoso está aumentando:**

| Ano | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dias ≥ 30 °C | 63 | 70 | 48 | 45 | 67 | 48 | 44 | 45 | 75 | **89** | 75 |
| Dias ≥ 32 °C | 21 | 13 | 4 | 7 | 22 | 11 | 4 | 4 | 27 | **36** | 20 |

  Média de 54 dias ≥ 30 °C por ano em 2015–2022 contra **80 por ano em 2023–2025**.

## Como os dados são usados no modelo

- **Pergunta:** *"Às 19h de hoje, dá para saber se amanhã o índice de calor máximo vai chegar a 30 °C ou mais?"*
- **Atributos:** calculados por `../atributos.py` a partir do CSV **horário**, usando só as horas de **00h a 18h** do dia D (o que existe às 19h, quando o modelo roda) e os dias anteriores.
- **Alvo:** `indice_calor_max_c` do **dia D+1** (dia inteiro) ≥ 30 °C.
- **Por que 30 °C e não 32 °C:** com 32 °C sobram só 86 dias positivos em 8 anos de treino, e nenhum modelo supera a regra "amanhã será como hoje" (análise na seção 4.4 do notebook). Além disso, a estação mede **menos** calor do que o interior de uma casa com telha de fibrocimento, então 30 °C na estação funciona como gatilho com margem de segurança.
- **Divisão treino/teste temporal (nunca aleatória):** treino 2015–2022 (429 dias perigosos), teste 2023–2025 (239 dias perigosos).

## Limitações (citadas no relatório)

- A estação do Mirante fica a 792 m de altitude, numa área aberta e ventilada. Por isso ela registra **menos calor** do que bairros densos, e muito menos do que o **interior de casas com telha de fibrocimento**. Isso justifica o dispositivo do HeatSafe medir o calor dentro da casa.
- O índice de calor é calculado a partir da temperatura e da umidade, não medido diretamente.
- 0,4% das horas e 22 dias estão sem dado.
- Uma única estação não "enxerga" sistemas que chegam de fora (frentes frias, massas de ar quente). Estações vizinhas do INMET são o próximo passo.
