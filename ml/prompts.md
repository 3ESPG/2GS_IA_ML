# Uso de IA como copiloto · IA & Machine Learning

Seguindo os combinados da GS (slide 21), registramos aqui como a IA foi usada nesta parte do HeatSafe. Todas as decisões foram revisadas pelo grupo, e cada integrante deve saber explicar cada etapa do notebook.

**Ferramenta:** Claude Code (Anthropic), em 09/11/2026.

## Prompt principal

> Preciso de ajuda para fazer um trabalho da faculdade. O PDF "GS CitizenTech - 3ES Fev - FIAP 2026" tem todas as informações, mas de outras matérias também; vamos focar apenas na matéria de Inteligência Artificial e Machine Learning (slides 31 a 33). Temos também os arquivos do nosso tema (PROPOSTA.md, "HeatSafe - Solução Escolhida" e "HeatSafe - Entrevistas (personas)") para usar e deixar com a nossa marca. Os dados do INMET estão na pasta `dados`. Veja o que deve ser feito e faça o trabalho completo. Integrantes: [nomes e RMs].

## O que a IA fez

| Etapa | Contribuição da IA | O que o grupo deve saber explicar |
|---|---|---|
| Leitura dos requisitos | Cruzou os slides 31–33 (entregáveis e rubrica) com a proposta e as personas | Os 6 critérios da nota e onde cada um aparece no notebook |
| Atributos | Escreveu `atributos.py` com a janela de 00h–18h (o modelo roda às 19h) | Por que usar o dia inteiro seria "espiar o futuro" |
| Experimentos | Testou XGBoost com vários ajustes e descobriu que, com o limiar de 32 °C, nenhum modelo superava a regra "amanhã = hoje" | A tabela da seção 4.4 e por que adotamos 30 °C |
| Escolha do modelo | Montou as duas grades (36 + 30 combinações) e a comparação dobra a dobra; o Random Forest venceu e substituiu o XGBoost planejado | Bagging × boosting e por que o Random Forest generalizou melhor aqui |
| Desbalanceamento | Comparou pesos de classe × ajuste do limiar | Por que o peso piora a probabilidade (Brier) |
| Avaliação | Métricas de treino × validação × teste, linhas de base, calibração, análise de erros | Por que acurácia engana; por que o ROC-AUC mostra que a validação foi honesta |
| Visualização | Gráficos com a paleta HeatSafe (laranja e azul, testados para daltonismo) | O que cada figura mostra |
| Textos | Rascunho do relatório, do README e do roteiro do vídeo | Toda a conclusão para o cidadão |

## Tentativa registrada

A IA tentou baixar dados de estações vizinhas do INMET (para o modelo "enxergar" frentes frias), mas o portal recusa downloads automatizados. Ficou como próximo passo no relatório: o grupo pode baixar os arquivos anuais manualmente em https://portal.inmet.gov.br/dadoshistoricos.

## Verificações feitas pelo grupo

- [ ] Rodar o notebook do zero (`Run All`) e conferir que os números batem com o relatório
- [ ] Cada integrante explicar em voz alta: a pergunta, a janela até 18h, a divisão temporal, a escolha do Random Forest, o limiar de 0,15 e a leitura para o cidadão
- [ ] Conferir a fonte e o período dos dados em `dados/README.md`
