# Water Quality Prediction

Este projeto tem como objetivo prever a potabilidade da água com base em várias características químicas. Utilizamos diferentes algoritmos de machine learning para treinar modelos e avaliar seu desempenho.

## Estrutura do Projeto

- `water_quality.py`: Script principal que contém o pipeline de machine learning.
- `Makefile`: Arquivo Makefile para facilitar a execução de tarefas comuns, como instalação de dependências e execução do script.
- `pyproject.toml`: Arquivo de configuração do Poetry para gerenciar dependências e configurações do projeto.

## Dependências

As principais bibliotecas utilizadas neste projeto são:

- `matplotlib`
- `mlflow`
- `numpy`
- `pandas`
- `scikit-learn`
- `poetry`

## Instalação

### Usando Poetry

1. Instale o Poetry, se ainda não estiver instalado:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

### Usando Makefile

1. Gera e atualiza as dependências do projeto usando o Makefile:
   ```bash
   make lock
   ```

2. Instale as dependências do projeto usando o Makefile:
   ```bash
   make install
   ```

## Execução

Para executar o script principal e treinar os modelos de machine learning, utilize o comando abaixo:

```bash
make run
```

## Estrutura do Código

O código está organizado em funções distintas para carregar dados, preparar dados, treinar e avaliar modelos, e registrar os resultados no MLflow. Isso facilita a manutenção e a reutilização do código.

### Funções Principais

- `load_data(file_path)`: Carrega os dados a partir de um arquivo CSV.
- `prepare_data(data)`: Prepara os dados para o treinamento, incluindo normalização e divisão em conjuntos de treino e teste.
- `train_and_evaluate(models, X_train, X_test, y_train, y_test)`: Treina e avalia os modelos de machine learning.
- `log_results(results)`: Registra os resultados no MLflow.

### Modelos Utilizados

Os seguintes modelos de machine learning são utilizados no projeto:

- Regressão Logística
- Árvore de Decisão
- Floresta Aleatória
- Gradient Boosting
- K-Nearest Neighbors
- Máquina de Vetores de Suporte
- Rede Neural

## Repositório
https://github.com/alwisnhe/mba-ml-iml4-ex02

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
