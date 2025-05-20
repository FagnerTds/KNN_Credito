# Classificação de Elegibilidade de Crédito com KNN

## Modelo escolhido
- KNN (K-Nearest Neighbors)

## Melhor valor de K
- K = 15 (exemplo, substitua pelo seu valor real)

```python
for k in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_previsto = knn.predict(X_test)
    acuracia = accuracy_score(y_test, y_previsto)
    if acuracia > melhor_acuracia:
        melhor_acuracia = acuracia
        melhor_k = k
```

## Features utilizadas 
- salário_anual
- total_dividas
- historico_pagamento (score)
- idade
- credito_solicitado

## Normalização
- Não foi utilizada, pois o modelo teve desempenho superior sem normalização.

## Exemplo de previsão
- Entrada: [10000, 5000, 0.92, 35, 3000]
- Saída: [2] → EElegível com Análise

## Classes
- 1 = Não Elegível
- 2 = Elegível com Análise
- 3 = Elegível

## Como usar?

```python
import joblib

# Carregar o modelo treinado
modelo = joblib.load('modelo_knn.joblib')

# Fazer uma previsão
entrada = [[10000, 5000, 0.92, 35, 3000]]  # Exemplo
saida = modelo.predict(entrada)

print(f"Resultado da previsão: {saida}")
```
