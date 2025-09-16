#  Neural Networks - Implementações Educacionais

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-orange.svg)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Educational](https://img.shields.io/badge/Purpose-Educational-purple.svg)](#)

Repositório com implementações educacionais de redes neurais em Python puro, focado no aprendizado dos conceitos fundamentais de Machine Learning e Deep Learning.

##  Objetivo

Este repositório foi criado para estudantes e entusiastas que desejam entender **como as redes neurais funcionam internamente**, implementando os algoritmos do zero usando apenas NumPy.

##  Estrutura do Projeto

```
Neural_Networks/
├──  mlp_simple.py              # Implementação base do MLP
├──  exemplos_mlp.py            # Exemplos práticos e casos de uso
├──  funcoes_ativacao.py        # Diferentes funções de ativação
├──  requirements.txt           # Dependências do projeto
├──  .gitignore                # Arquivos ignorados pelo Git
└──  README.md                 # Este arquivo
```

##  Implementações Disponíveis

### 1. **MLP Básico** (`mlp_simple.py`)
-  Forward propagation
-  Backpropagation
-  Gradient descent
-  Função de ativação Sigmoid
-  Classificação e regressão

### 2. **Funções de Ativação** (`funcoes_ativacao.py`)
-  Sigmoid
-  Tanh
-  ReLU
-  Leaky ReLU
-  ELU
-  Swish

### 3. **Exemplos Práticos** (`exemplos_mlp.py`)
-  Problema XOR
-  Classificação linear
-  Círculos concêntricos
-  Regressão não-linear
-  Comparação de arquiteturas

## Como Usar

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Executar o Exemplo Básico (XOR)

```bash
python mlp_simple.py
```

### 3. Executar Exemplos Avançados

```bash
python exemplos_mlp.py
```

## Conceitos Implementados

### Arquitetura da Rede
- **Camadas totalmente conectadas**: Cada neurônio se conecta a todos os neurônios da próxima camada
- **Função de ativação Sigmoid**: Usada em todas as camadas
- **Inicialização Xavier/Glorot**: Para melhor convergência

### Algoritmos
- **Forward Pass**: Propagação dos dados da entrada até a saída
- **Backpropagation**: Cálculo dos gradientes e propagação do erro
- **Gradient Descent**: Atualização dos pesos baseada nos gradientes

### Funcionalidades
-  Classificação binária
-  Regressão
-  Múltiplas camadas ocultas
-  Visualização de resultados
-  Métricas de avaliação

## Exemplos Incluídos

### 1. Problema XOR (Básico)
```python
from mlp_simple import MLPSimples
import numpy as np

# Dados XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Criar e treinar rede
mlp = MLPSimples(camadas=[2, 4, 1], taxa_aprendizado=0.5)
mlp.treinar(X, y, epocas=2000)

# Fazer previsões
previsoes = mlp.prever(X)
print(previsoes)
```

### 2. Classificação Linear
- Dados sintéticos com 2 features
- Separação linear de classes
- Avaliação com conjunto de teste

### 3. Círculos Concêntricos (Não-linear)
- Problema mais complexo que requer múltiplas camadas
- Visualização da fronteira de decisão
- Demonstra a capacidade de aprender padrões não-lineares

### 4. Regressão (Função Seno)
- Aproximação de função contínua
- Demonstra uso da rede para regressão
- Visualização da curva aprendida

### 5. Comparação de Arquiteturas
- Testa diferentes números de camadas e neurônios
- Compara performance no problema XOR
- Ajuda a entender o impacto da arquitetura

## Parâmetros da Rede

### Construtor da MLPSimples
```python
MLPSimples(camadas, taxa_aprendizado=0.01)
```

- **`camadas`**: Lista com número de neurônios por camada
  - Exemplo: `[2, 4, 3, 1]` = 2 entradas, 4 neurônios na 1ª camada oculta, 3 na 2ª, 1 saída
- **`taxa_aprendizado`**: Controla o tamanho dos passos na atualização dos pesos
  - Valores típicos: 0.001 a 1.0
  - Maior = aprendizado mais rápido, mas pode ser instável
  - Menor = aprendizado mais estável, mas mais lento

### Método de Treinamento
```python
treinar(X, y, epocas=1000, verbose=True)
```

- **`X`**: Dados de entrada (matriz numpy)
- **`y`**: Rótulos/targets (matriz numpy)
- **`epocas`**: Número de iterações de treinamento
- **`verbose`**: Se deve imprimir progresso

## Interpretando os Resultados

### Curva de Aprendizado
- **Decrescente**: Rede está aprendendo
- **Estável**: Convergiu ou precisa de mais épocas
- **Oscilante**: Taxa de aprendizado pode estar muito alta

### Métricas
- **Acurácia**: % de previsões corretas (classificação)
- **MSE**: Erro médio quadrático (regressão)
- **Diferença treino/teste**: Indica overfitting se muito grande

## Dicas para Experimentar

### Ajuste de Hiperparâmetros
1. **Taxa de Aprendizado**:
   - Comece com 0.1
   - Se não converge: diminua (0.01, 0.001)
   - Se muito lento: aumente (0.5, 1.0)

2. **Arquitetura**:
   - Problemas simples: 1-2 camadas ocultas
   - Problemas complexos: 3-4 camadas ocultas
   - Mais neurônios = mais capacidade, mas risco de overfitting

3. **Épocas**:
   - Monitore a curva de erro
   - Pare quando estabilizar
   - Típico: 1000-5000 épocas

