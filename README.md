# Rede Neural Convolucional (CNN)

Implementação de uma rede neural convolucional sobre o dataset MNIST, suportando classificação binária ou multiclasse (padrão).

Projeto criado como Exercício Programa da disciplina de Inteligência Artificial do Curso de Sistemas de Informação da USP.

## Descrição

Este projeto consiste na implementação de uma rede neural convolucional (CNN) para a classificação de imagens do dataset MNIST. O modelo pode ser configurado para realizar classificação binária ou multiclasse (10 classes).

## Requisitos

Para executar este projeto, é necessário instalar os pacotes listados no arquivo `requirements.txt`. Utilize o seguinte comando para instalar as dependências:

```bash
pip install -r requirements.txt
```

## Uso

Clone o repositório e rode a entrada para API usando:
```bash
python entrypoint.py
```
Crie requisições tipo POST para a API.
Um exemplo de utilização foi disponibilizado em `API/exemplo.postman_collection.json`.


## Estrutura do Projeto

### Diretório `Models`

- `pesos_iniciais.weights.h5`: Pesos iniciais do modelo.
- `pesos_finais.h5`: Pesos finais do modelo após o treinamento.
- `modelo_final.keras`: Modelo final salvo.
- `hyperparameters.json`: Arquivo JSON contendo os hiperparâmetros utilizados e informações das camadas da CNN.

### Diretório `Outputs`

- `predictions.csv`: Arquivo CSV com as previsões do modelo para o conjunto de teste.
- `loss_history.txt`: Arquivo de texto contendo o erro e acurácia a cada época do treinamento.
- `matriz_confusao.png`: Imagem da matriz de confusão.

### Diretório `scripts`

- `load_data.py`: Script para carregamento e pré-processamento dos dados.
- `train_model.py`: Script para treinamento modelo.
- `evaluate_model.py`: Script para avaliação do modelo.


