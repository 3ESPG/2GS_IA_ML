![HeatSafe](figuras/banner.png)

# Relatório de Machine Learning · HeatSafe

**GS 2026.2 · CitizenTech · Inteligência Artificial & Machine Learning** · FIAP, Engenharia de Software, 3º ano (Fev)
**Integrantes:** Felipe Braunstein e Silva (RM554483) · Felipe do Nascimento Fernandes (RM554598) · Henrique Ignacio Bartalo (RM555274) · Gustavo Henrique Martins (RM556956)
**Notebook executado:** [`heatsafe_previsao_dia_perigoso.ipynb`](heatsafe_previsao_dia_perigoso.ipynb) · **Dados:** [`dados/README.md`](dados/README.md)

---

## 1. Problema e pergunta de ML

O **HeatSafe** é um dispositivo instalado na casa de idosos que passam o dia sozinhos. Ele mede o calor dentro da casa e percebe se a pessoa está se movimentando, e avisa a família ou a UBS quando a situação fica perigosa. Nas entrevistas, a Dona Maria (78 anos, casa com telha de fibrocimento) contou que *"não sente sede"* e já desmaiou sozinha no verão. A agente de saúde Rosa disse que *"no dia de calor não sei quem está pior naquele momento"*.

A parte de ML dá ao HeatSafe **um dia de antecedência**: na véspera, às 19h, o modelo diz se amanhã será um dia perigoso. Se for, liga o **modo vigilância** (mais lembretes de água, família e UBS avisadas antes).

> **Pergunta:** às 19h de hoje, com o que a estação meteorológica já mediu, dá para saber se **amanhã o índice de calor máximo em São Paulo vai chegar a 30 °C ou mais?** (classificação binária)

**Custo dos erros:** deixar passar um dia perigoso (falso negativo) é o pior erro, porque deixa a Dona Maria desprotegida. Um alarme falso custa pouco, porque o alerta à família só dispara se o sensor **dentro da casa** confirmar o calor e a falta de movimento. Por isso otimizamos o **F2**, que dá peso 2 ao recall, sem ignorar a precisão: a filha, Ana, avisou que *"se o celular apitar toda hora, vou acabar ignorando"*.

## 2. Dados reais e públicos

| Fonte | Estação | Período | Volume |
|---|---|---|---|
| **INMET · BDMEP** ([portal.inmet.gov.br/dadoshistoricos](https://portal.inmet.gov.br/dadoshistoricos)), baixado em 08/11/2026 | **A701, São Paulo (Mirante de Santana)**, 792 m | **01/01/2015 a 31/12/2025** | **96.430 medições horárias** (4.018 dias) |

Tratamento (`dados/preparar_dados.py`): leitura dos 11 CSVs anuais, `-9999` → vazio (0,4% das horas), conversão de UTC para o horário de Brasília e cálculo do **índice de calor da NOAA** hora a hora (a mesma fórmula do ESP32). Nenhum dado é simulado. Os dados nos mostram uma tendência clara: média de **54 dias ≥ 30 °C por ano em 2015–2022** contra **80 por ano em 2023–2025**.

## 3. Preparação e divisão treino/teste

- **Sem espiar o futuro:** como o modelo roda às 19h, os atributos do dia D usam **só as medições de 00h a 18h**. O alvo é o índice de calor máximo do **dia D+1 inteiro**.
- **30 atributos** (`atributos.py`, o mesmo código usado em produção): o calor, a umidade, o ponto de orvalho, a pressão, a radiação, a chuva e o vento do dia D; os valores de D-1 e D-2; médias de 3 e 7 dias; variação de ontem para hoje; e a época do ano (seno e cosseno do dia do ano).
- **Faltantes:** preenchidos com a mediana **do treino**, dentro do *pipeline* (recalculada a cada dobra).
- **Divisão temporal, nunca aleatória:** treino com os dias previstos de **2015 a 2022** (2.914 dias, 429 perigosos, 14,7%) e teste com **2023 a 2025** (1.081 dias, 239 perigosos, 22,1%). O teste foi usado **uma única vez**, no final.
- **Validação cruzada temporal:** `TimeSeriesSplit` com 5 dobras, só no treino. Todas as escolhas (limiar do alvo, modelo, hiperparâmetros, limiar de decisão) saíram dela.

**Por que 30 °C e não 32 °C?** Testamos os dois na validação cruzada. Com 32 °C sobram só 86 dias positivos em 8 anos, e nenhum modelo supera a regra "amanhã será como hoje" (AP de 0,30 contra 0,31). Com 30 °C os modelos passam a superá-la (AP de 0,65 contra 0,60). Além disso, a estação fica num morro ventilado e mede menos calor do que uma casa de fibrocimento, então 30 °C na estação funciona como gatilho com margem de segurança.

## 4. Escolha do modelo

Os dados são uma tabela pequena (≈ 2.900 dias) com relações não lineares. Por isso levamos dois comitês de árvores até o fim e deixamos a validação cruzada decidir:

- **XGBoost** (boosting): árvores em sequência, cada uma corrige a anterior. Tem baixo viés, mas corre risco de decorar com poucos positivos.
- **Random Forest** (bagging): média de centenas de árvores independentes. Tem baixa variância e é estável com rótulos "no limite" (29,9 °C × 30,1 °C).

Descartamos **LSTM** (pouco dado; a memória temporal já entra nos atributos D-1, D-2 e nas médias), **KMeans + PCA** (é não supervisionado, e nós temos o rótulo) e **MLP** (sensível a hiperparâmetros e pouco interpretável com tão poucos positivos).

**Linhas de base que o modelo precisa superar:** "nunca avisar" (78% de acurácia e recall zero, por isso **não usamos a acurácia**) e a **persistência**: "se hoje o índice passou de *t*, amanhã também passa", com *t* = 28,6 °C ajustado no treino.

## 5. Treino com diferentes hiperparâmetros

| Modelo | Hiperparâmetros (3) | Combinações | AP na validação (pior → melhor) | Melhor combinação |
|---|---|---|---|---|
| XGBoost | `max_depth` {1,2,3,4} · `learning_rate` {0,01; 0,03; 0,1} · `n_estimators` {100,300,600} | **36** | 0,557 → **0,645** | depth 2, lr 0,03, 100 árvores |
| Random Forest | `max_depth` {4,6,8,12,∞} · `min_samples_leaf` {1,5,20} · `max_features` {√, 50%} | **30** | 0,640 → **0,668** | depth 8, folha 5, √ atributos |

Cada combinação foi avaliada nas 5 dobras temporais (330 treinos). A métrica de escolha foi o **Average Precision** (área sob a curva precisão × recall), adequado para classes desbalanceadas. O **Random Forest venceu**: AP médio maior (0,668 × 0,645), melhor em 3 de 5 dobras e com probabilidades mais confiáveis (Brier 0,070 × 0,072). O XGBoost era o plano inicial do grupo; **trocamos porque os dados mostraram que o Random Forest generaliza melhor.**

Testamos dar mais peso à classe rara (`class_weight="balanced"`). O AP não melhorou e as probabilidades pioraram (Brier de 0,070 para 0,083). Mantivemos o treino sem peso e tratamos o desbalanceamento pelo **limiar de decisão**, escolhido nas previsões fora da dobra pelo maior F2: **o modo vigilância liga quando P(amanhã perigoso) ≥ 0,15**.

## 6. Resultados em treino e teste

| Random Forest | Recall | Precisão | F2 | AP | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| Treino (2015–2022) | 0,988 | 0,527 | 0,841 | 0,923 | 0,983 | 0,042 |
| Validação cruzada (fora da dobra) | 0,894 | 0,409 | 0,723 | 0,654 | 0,921 | 0,070 |
| **Teste (2023–2025)** | **0,904** | **0,522** | **0,788** | **0,774** | **0,917** | **0,095** |

![Matrizes de confusão em treino e teste](figuras/fig06_matrizes_de_confusao.png)

**Overfitting.** O Random Forest quase decora o treino (recall de 99%, AP de 0,92), porque cada árvore já viu a maior parte desses dias. Por isso o número de treino não serve como estimativa, e quem cumpre esse papel é a validação cruzada. A prova de que ela funcionou é o **ROC-AUC, que não depende da proporção de positivos: 0,921 na validação e 0,917 no teste**. O AP e a precisão *subiram* no teste porque 2023–2025 teve mais dias perigosos (22% contra 15%), e não porque o modelo melhorou. A busca controlou o overfitting escolhendo folhas com pelo menos 5 dias e profundidade 8 (figura 4 do notebook).

| Teste 2023–2025 (239 dias perigosos) | Recall | Precisão | F2 | AP | Avisados | Perdidos | Alarmes falsos |
|---|---|---|---|---|---|---|---|
| Nunca avisar | 0,000 | — | 0,000 | 0,221 | 0 | 239 | 0 |
| Persistência (hoje ≥ 28,6 °C) | 0,883 | 0,551 | 0,788 | 0,713 | 211 | 28 | 172 |
| XGBoost | 0,782 | 0,607 | 0,740 | 0,752 | 187 | 52 | 121 |
| **Random Forest (HeatSafe)** | **0,904** | 0,522 | **0,788** | **0,774** | **216** | **23** | 198 |

![Curva precisão x recall no teste](figuras/fig07_curva_precisao_recall_teste.png)

**Leitura honesta:** a persistência é uma linha de base forte, porque o calor de hoje é o atributo mais importante do modelo. O Random Forest **ordena melhor** os dias por risco (maior AP e ROC-AUC), avisa **5 dias perigosos a mais** com **26 alarmes falsos a mais em 3 anos** e entrega uma **probabilidade calibrada** (Brier 0,095 contra 0,178 da referência). A regra simples não oferece essa probabilidade. O XGBoost faz menos alarmes falsos, mas perde 52 dias perigosos, mais que o dobro.

**Onde o modelo erra:** dos 23 dias perdidos, 21 ficaram entre 30,0 e 31,5 °C, logo acima do limiar. Dos **83 dias mais extremos** (≥ 32 °C), avisou **81 (98%)**. Os alarmes falsos foram dias quentes, não frios: a mediana foi de 28,1 °C, e 65% estavam no nível *Atenção*.

![Previsões no período de teste](figuras/fig10_calendario_teste.png)

## 7. Conclusão: o que o resultado significa para o cidadão

- **Para a Dona Maria:** em **9 de cada 10 dias de calor perigoso**, o HeatSafe entra em vigilância na véspera, e os lembretes de água começam antes de a casa "virar um forno". Na **onda de calor de novembro de 2023** (recorde de 35,5 °C), **10 de 11 dias** foram avisados na véspera.
- **Para a Ana:** a probabilidade permite **dois níveis**. Com P ≥ 0,15 o modo vigilância liga em silêncio (só o dispositivo e a UBS). Com **P ≥ 0,50** ela é avisada na véspera, e **79% desses avisos se confirmam** (cerca de 56 por ano, concentrados no verão). Isso responde ao pedido de *"só ser avisada quando for sério"*.
- **Para a Rosa:** às 19h da véspera, a UBS já sabe que amanhã é dia de vigilância e pode reorganizar as visitas.

**Limitações:** (1) uma única estação não enxerga a frente fria ou a massa de ar quente que chega de fora, por isso o ganho sobre a regra simples é real, porém modesto; (2) o limiar de 30 °C é uma escolha de projeto que ainda precisa ser calibrada com a temperatura dentro das casas; (3) o clima mudou dentro dos próprios dados, e o modelo ficou levemente conservador em 2023–2025 (previu 19% de dias perigosos, contra 22% reais). Por isso o **modelo de produção é retreinado com 2015–2025** (`modelo/heatsafe_modelo.joblib`, consumido pelo serviço de previsão via `prever.py`).

**Próximos passos:** incluir estações vizinhas do INMET (Curitiba, Sorocaba); calibrar o limiar com os dados dos próprios dispositivos HeatSafe; e fazer o retreino anual automático no serviço.

**ODS:** **3** (meta 3.d, alerta precoce de riscos à saúde) · **13** (meta 13.1, adaptação ao aumento dos dias de calor extremo, de 54 para 80 por ano) · **11** (cidades resilientes, cuidado orientado por bairro) · **10** (protege quem menos usa tecnologia, sem exigir que o idoso use aplicativo).
