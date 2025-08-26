import numpy as np
import matplotlib.pyplot as plt
from mlp_simple import MLPSimples
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def exemplo_classificacao_linear():
    """
    Exemplo com um problema de classificação linear simples.
    """
    print("=== Exemplo: Classificação Linear ===")
    
    # Gerar dados sintéticos
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, 
                             random_state=42)
    
    # Normalizar os dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape y para formato correto
    y = y.reshape(-1, 1)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Criar e treinar a rede
    mlp = MLPSimples(camadas=[2, 8, 4, 1], taxa_aprendizado=0.1)
    
    print("Treinando a rede...")
    historico_erro = mlp.treinar(X_train, y_train, epocas=1000, verbose=True)
    
    # Avaliar no conjunto de teste
    acuracia_treino = mlp.avaliar_acuracia(X_train, y_train)
    acuracia_teste = mlp.avaliar_acuracia(X_test, y_test)
    
    print(f"\nAcurácia no treino: {acuracia_treino:.2%}")
    print(f"Acurácia no teste: {acuracia_teste:.2%}")
    
    # Plotar resultados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Curva de aprendizado
    ax1.plot(historico_erro)
    ax1.set_title('Curva de Aprendizado')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Erro (MSE)')
    ax1.grid(True)
    
    # Visualizar classificação
    scatter = ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test.ravel(), 
                         cmap='viridis', alpha=0.7)
    ax2.set_title('Dados de Teste')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax2)
    
    plt.tight_layout()
    plt.show()

def exemplo_circulos_concentricos():
    """
    Exemplo com problema não-linear (círculos concêntricos).
    """
    print("\n=== Exemplo: Círculos Concêntricos (Não-linear) ===")
    
    # Gerar dados de círculos concêntricos
    X, y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)
    
    # Reshape y para formato correto
    y = y.reshape(-1, 1)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Criar e treinar a rede (mais camadas para problema não-linear)
    mlp = MLPSimples(camadas=[2, 10, 8, 6, 1], taxa_aprendizado=0.3)
    
    print("Treinando a rede...")
    historico_erro = mlp.treinar(X_train, y_train, epocas=2000, verbose=True)
    
    # Avaliar no conjunto de teste
    acuracia_treino = mlp.avaliar_acuracia(X_train, y_train)
    acuracia_teste = mlp.avaliar_acuracia(X_test, y_test)
    
    print(f"\nAcurácia no treino: {acuracia_treino:.2%}")
    print(f"Acurácia no teste: {acuracia_teste:.2%}")
    
    # Plotar resultados
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Curva de aprendizado
    axes[0, 0].plot(historico_erro)
    axes[0, 0].set_title('Curva de Aprendizado')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Erro (MSE)')
    axes[0, 0].grid(True)
    
    # Dados originais
    scatter1 = axes[0, 1].scatter(X[:, 0], X[:, 1], c=y.ravel(), 
                                 cmap='viridis', alpha=0.7)
    axes[0, 1].set_title('Dados Originais')
    axes[0, 1].set_xlabel('X1')
    axes[0, 1].set_ylabel('X2')
    plt.colorbar(scatter1, ax=axes[0, 1])
    
    # Previsões no conjunto de teste
    previsoes_teste = mlp.prever(X_test)
    scatter2 = axes[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=previsoes_teste.ravel(), 
                                 cmap='viridis', alpha=0.7)
    axes[1, 0].set_title('Previsões da Rede (Teste)')
    axes[1, 0].set_xlabel('X1')
    axes[1, 0].set_ylabel('X2')
    plt.colorbar(scatter2, ax=axes[1, 0])
    
    # Fronteira de decisão
    plotar_fronteira_decisao(mlp, X, y, axes[1, 1])
    
    plt.tight_layout()
    plt.show()

def plotar_fronteira_decisao(mlp, X, y, ax):
    """
    Plota a fronteira de decisão da rede neural.
    """
    h = 0.02  # Tamanho do passo na malha
    
    # Criar uma malha
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Fazer previsões na malha
    malha_pontos = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.prever(malha_pontos)
    Z = Z.reshape(xx.shape)
    
    # Plotar a fronteira de decisão
    ax.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='viridis')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), 
                        cmap='viridis', edgecolors='black')
    ax.set_title('Fronteira de Decisão')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

def exemplo_regressao():
    """
    Exemplo de regressão com função seno.
    """
    print("\n=== Exemplo: Regressão (Função Seno) ===")
    
    # Gerar dados de uma função seno com ruído
    X = np.linspace(0, 4*np.pi, 200).reshape(-1, 1)
    y = np.sin(X) + 0.1 * np.random.randn(200, 1)
    
    # Normalizar entrada
    X_norm = (X - X.mean()) / X.std()
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)
    X_train_orig, X_test_orig, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Criar e treinar a rede
    mlp = MLPSimples(camadas=[1, 20, 15, 10, 1], taxa_aprendizado=0.01)
    
    print("Treinando a rede...")
    historico_erro = mlp.treinar(X_train, y_train, epocas=3000, verbose=True)
    
    # Fazer previsões
    previsoes_treino = mlp.prever(X_train)
    previsoes_teste = mlp.prever(X_test)
    
    # Calcular erro médio quadrático
    mse_treino = np.mean((previsoes_treino - y_train) ** 2)
    mse_teste = np.mean((previsoes_teste - y_test) ** 2)
    
    print(f"\nMSE no treino: {mse_treino:.4f}")
    print(f"MSE no teste: {mse_teste:.4f}")
    
    # Plotar resultados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Curva de aprendizado
    ax1.plot(historico_erro)
    ax1.set_title('Curva de Aprendizado')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Erro (MSE)')
    ax1.grid(True)
    
    # Resultados da regressão
    ax2.scatter(X_train_orig, y_train, alpha=0.6, label='Treino', color='blue')
    ax2.scatter(X_test_orig, y_test, alpha=0.6, label='Teste', color='red')
    ax2.scatter(X_test_orig, previsoes_teste, alpha=0.8, label='Previsões', color='green', marker='x')
    ax2.set_title('Regressão - Função Seno')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def comparar_arquiteturas():
    """
    Compara diferentes arquiteturas de rede no problema XOR.
    """
    print("\n=== Comparação de Arquiteturas (XOR) ===")
    
    # Dados XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Diferentes arquiteturas para testar
    arquiteturas = {
        'Pequena': [2, 3, 1],
        'Média': [2, 4, 1],
        'Grande': [2, 8, 4, 1],
        'Muito Grande': [2, 10, 8, 4, 1]
    }
    
    resultados = {}
    
    plt.figure(figsize=(15, 10))
    
    for i, (nome, arquitetura) in enumerate(arquiteturas.items()):
        print(f"\nTestando arquitetura {nome}: {arquitetura}")
        
        mlp = MLPSimples(camadas=arquitetura, taxa_aprendizado=0.5)
        historico_erro = mlp.treinar(X, y, epocas=2000, verbose=False)
        
        acuracia = mlp.avaliar_acuracia(X, y)
        erro_final = historico_erro[-1]
        
        resultados[nome] = {
            'acuracia': acuracia,
            'erro_final': erro_final,
            'historico': historico_erro
        }
        
        print(f"Acurácia: {acuracia:.2%}, Erro final: {erro_final:.6f}")
        
        # Plotar curva de aprendizado
        plt.subplot(2, 2, i+1)
        plt.plot(historico_erro)
        plt.title(f'{nome}: {arquitetura}')
        plt.xlabel('Época')
        plt.ylabel('Erro (MSE)')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Resumo dos resultados
    print("\n=== Resumo dos Resultados ===")
    for nome, resultado in resultados.items():
        print(f"{nome}: Acurácia = {resultado['acuracia']:.2%}, "
              f"Erro Final = {resultado['erro_final']:.6f}")

if __name__ == "__main__":
    # Executar todos os exemplos
    exemplo_classificacao_linear()
    exemplo_circulos_concentricos()
    exemplo_regressao()
    comparar_arquiteturas()