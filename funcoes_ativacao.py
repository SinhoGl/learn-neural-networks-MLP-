import numpy as np
import matplotlib.pyplot as plt
from mlp_simple import MLPSimples

class MLPComFuncoesAtivacao(MLPSimples):
    """
    Extensão da classe MLPSimples com diferentes funções de ativação.
    """
    
    def __init__(self, camadas, taxa_aprendizado=0.01, funcao_ativacao='sigmoid'):
        """
        Inicializa a rede MLP com função de ativação escolhida.
        
        Args:
            camadas: Lista com o número de neurônios em cada camada
            taxa_aprendizado: Taxa de aprendizado para o treinamento
            funcao_ativacao: Tipo de função de ativação ('sigmoid', 'tanh', 'relu', 'leaky_relu')
        """
        super().__init__(camadas, taxa_aprendizado)
        self.funcao_ativacao = funcao_ativacao
    
    # ==================== FUNÇÕES DE ATIVAÇÃO ====================
    
    def sigmoid(self, x):
        """Função Sigmoid: f(x) = 1 / (1 + e^(-x))
        
        Características:
        - Saída entre 0 e 1
        - Suave e diferenciável
        - Problema: Vanishing gradient para valores extremos
        - Uso: Camada de saída para classificação binária
        """
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def derivada_sigmoid(self, x):
        """Derivada da sigmoid: f'(x) = f(x) * (1 - f(x))"""
        return x * (1 - x)
    
    def tanh(self, x):
        """Função Tangente Hiperbólica: f(x) = tanh(x)
        
        Características:
        - Saída entre -1 e 1
        - Centrada em zero (melhor que sigmoid)
        - Ainda sofre de vanishing gradient
        - Uso: Camadas ocultas, melhor que sigmoid
        """
        return np.tanh(x)
    
    def derivada_tanh(self, x):
        """Derivada da tanh: f'(x) = 1 - tanh²(x)"""
        return 1 - x**2
    
    def relu(self, x):
        """Função ReLU (Rectified Linear Unit): f(x) = max(0, x)
        
        Características:
        - Saída entre 0 e +∞
        - Computacionalmente eficiente
        - Resolve problema do vanishing gradient
        - Problema: Neurônios "mortos" (dying ReLU)
        - Uso: Camadas ocultas (mais popular atualmente)
        """
        return np.maximum(0, x)
    
    def derivada_relu(self, x):
        """Derivada da ReLU: f'(x) = 1 se x > 0, senão 0"""
        return (x > 0).astype(float)
    
    def leaky_relu(self, x, alpha=0.01):
        """Função Leaky ReLU: f(x) = max(αx, x)
        
        Características:
        - Saída entre -∞ e +∞
        - Resolve o problema dos neurônios "mortos"
        - Pequeno gradiente para valores negativos
        - Uso: Alternativa ao ReLU quando há neurônios mortos
        """
        return np.where(x > 0, x, alpha * x)
    
    def derivada_leaky_relu(self, x, alpha=0.01):
        """Derivada da Leaky ReLU: f'(x) = 1 se x > 0, senão α"""
        return np.where(x > 0, 1, alpha)
    
    def elu(self, x, alpha=1.0):
        """Função ELU (Exponential Linear Unit): f(x) = x se x > 0, senão α(e^x - 1)
        
        Características:
        - Saída entre -α e +∞
        - Suave em todos os pontos
        - Média das ativações próxima de zero
        - Uso: Alternativa ao ReLU com melhor performance
        """
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def derivada_elu(self, x, alpha=1.0):
        """Derivada da ELU: f'(x) = 1 se x > 0, senão α * e^x"""
        return np.where(x > 0, 1, alpha * np.exp(x))
    
    def swish(self, x):
        """Função Swish: f(x) = x * sigmoid(x)
        
        Características:
        - Saída entre -∞ e +∞
        - Suave e não-monotônica
        - Boa performance em redes profundas
        - Uso: Alternativa moderna ao ReLU
        """
        return x * self.sigmoid(x)
    
    def derivada_swish(self, x):
        """Derivada da Swish: f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))"""
        sig = self.sigmoid(x)
        return sig + x * sig * (1 - sig)
    
    # ==================== APLICAÇÃO DAS FUNÇÕES ====================
    
    def aplicar_ativacao(self, x):
        """Aplica a função de ativação escolhida"""
        if self.funcao_ativacao == 'sigmoid':
            return self.sigmoid(x)
        elif self.funcao_ativacao == 'tanh':
            return self.tanh(x)
        elif self.funcao_ativacao == 'relu':
            return self.relu(x)
        elif self.funcao_ativacao == 'leaky_relu':
            return self.leaky_relu(x)
        elif self.funcao_ativacao == 'elu':
            return self.elu(x)
        elif self.funcao_ativacao == 'swish':
            return self.swish(x)
        else:
            raise ValueError(f"Função de ativação '{self.funcao_ativacao}' não suportada")
    
    def aplicar_derivada(self, x):
        """Aplica a derivada da função de ativação escolhida"""
        if self.funcao_ativacao == 'sigmoid':
            return self.derivada_sigmoid(x)
        elif self.funcao_ativacao == 'tanh':
            return self.derivada_tanh(x)
        elif self.funcao_ativacao == 'relu':
            return self.derivada_relu(x)
        elif self.funcao_ativacao == 'leaky_relu':
            return self.derivada_leaky_relu(x)
        elif self.funcao_ativacao == 'elu':
            return self.derivada_elu(x)
        elif self.funcao_ativacao == 'swish':
            return self.derivada_swish(x)
        else:
            raise ValueError(f"Função de ativação '{self.funcao_ativacao}' não suportada")
    
    # ==================== OVERRIDE DOS MÉTODOS PRINCIPAIS ====================
    
    def forward_pass(self, X):
        """Forward pass usando a função de ativação escolhida"""
        ativacoes = [X]
        z_valores = []
        
        entrada_atual = X
        
        for i in range(self.num_camadas - 1):
            # Calcular z = X * W + b
            z = np.dot(entrada_atual, self.pesos[i]) + self.bias[i]
            z_valores.append(z)
            
            # Aplicar função de ativação
            ativacao = self.aplicar_ativacao(z)
            ativacoes.append(ativacao)
            
            entrada_atual = ativacao
        
        return ativacoes, z_valores
    
    def backward_pass(self, X, y, ativacoes):
        """Backward pass usando a derivada da função de ativação escolhida"""
        m = X.shape[0]
        
        # Calcular erro da camada de saída
        erro_saida = ativacoes[-1] - y
        
        # Lista para armazenar os gradientes
        gradientes_pesos = []
        gradientes_bias = []
        
        # Backpropagation
        erro_atual = erro_saida
        
        for i in range(self.num_camadas - 2, -1, -1):
            # Gradiente dos pesos
            grad_peso = np.dot(ativacoes[i].T, erro_atual) / m
            grad_bias = np.mean(erro_atual, axis=0, keepdims=True)
            
            gradientes_pesos.insert(0, grad_peso)
            gradientes_bias.insert(0, grad_bias)
            
            # Calcular erro para a camada anterior (se não for a primeira)
            if i > 0:
                erro_atual = np.dot(erro_atual, self.pesos[i].T) * self.aplicar_derivada(ativacoes[i])
        
        # Atualizar pesos e bias
        for i in range(self.num_camadas - 1):
            self.pesos[i] -= self.taxa_aprendizado * gradientes_pesos[i]
            self.bias[i] -= self.taxa_aprendizado * gradientes_bias[i]


def visualizar_funcoes_ativacao():
    """
    Visualiza as diferentes funções de ativação e suas derivadas.
    """
    x = np.linspace(-5, 5, 1000)
    
    # Criar instância para acessar as funções
    mlp = MLPComFuncoesAtivacao([1, 1])
    
    # Calcular as funções
    funcoes = {
        'Sigmoid': (mlp.sigmoid(x), mlp.derivada_sigmoid(mlp.sigmoid(x))),
        'Tanh': (mlp.tanh(x), mlp.derivada_tanh(mlp.tanh(x))),
        'ReLU': (mlp.relu(x), mlp.derivada_relu(x)),
        'Leaky ReLU': (mlp.leaky_relu(x), mlp.derivada_leaky_relu(x)),
        'ELU': (mlp.elu(x), mlp.derivada_elu(x)),
        'Swish': (mlp.swish(x), mlp.derivada_swish(x))
    }
    
    # Plotar as funções
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, (nome, (funcao, derivada)) in enumerate(funcoes.items()):
        # Função de ativação
        axes[i].plot(x, funcao, 'b-', linewidth=2, label=f'{nome}')
        axes[i].plot(x, derivada, 'r--', linewidth=2, label=f"Derivada {nome}")
        axes[i].set_title(f'{nome}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('f(x)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        axes[i].set_ylim(-2, 2)
    
    plt.tight_layout()
    plt.show()


def comparar_funcoes_ativacao_xor():
    """
    Compara diferentes funções de ativação no problema XOR.
    """
    print("=== Comparação de Funções de Ativação - Problema XOR ===")
    
    # Dados XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Funções de ativação para testar
    funcoes = ['sigmoid', 'tanh', 'relu', 'leaky_relu']
    
    resultados = {}
    
    plt.figure(figsize=(15, 10))
    
    for i, funcao in enumerate(funcoes):
        print(f"\nTestando função: {funcao.upper()}")
        
        try:
            # Ajustar taxa de aprendizado para cada função
            if funcao in ['relu', 'leaky_relu']:
                taxa = 0.01  # ReLU precisa de taxa menor
            else:
                taxa = 0.5
            
            mlp = MLPComFuncoesAtivacao(camadas=[2, 8, 4, 1], 
                                      taxa_aprendizado=taxa, 
                                      funcao_ativacao=funcao)
            
            historico_erro = mlp.treinar(X, y, epocas=3000, verbose=False)
            
            acuracia = mlp.avaliar_acuracia(X, y)
            erro_final = historico_erro[-1]
            
            resultados[funcao] = {
                'acuracia': acuracia,
                'erro_final': erro_final,
                'historico': historico_erro
            }
            
            print(f"Acurácia: {acuracia:.2%}, Erro final: {erro_final:.6f}")
            
            # Plotar curva de aprendizado
            plt.subplot(2, 2, i+1)
            plt.plot(historico_erro)
            plt.title(f'{funcao.upper()}: Acurácia = {acuracia:.1%}')
            plt.xlabel('Época')
            plt.ylabel('Erro (MSE)')
            plt.grid(True)
            
        except Exception as e:
            print(f"Erro ao treinar com {funcao}: {e}")
            resultados[funcao] = {'erro': str(e)}
    
    plt.tight_layout()
    plt.show()
    
    # Resumo dos resultados
    print("\n=== Resumo dos Resultados ===")
    for funcao, resultado in resultados.items():
        if 'erro' in resultado:
            print(f"{funcao.upper()}: ERRO - {resultado['erro']}")
        else:
            print(f"{funcao.upper()}: Acurácia = {resultado['acuracia']:.2%}, "
                  f"Erro Final = {resultado['erro_final']:.6f}")


def exemplo_regressao_com_diferentes_ativacoes():
    """
    Testa diferentes funções de ativação em um problema de regressão.
    """
    print("\n=== Regressão com Diferentes Funções de Ativação ===")
    
    # Gerar dados de uma função não-linear
    X = np.linspace(-2, 2, 200).reshape(-1, 1)
    y = X**3 - 2*X**2 + X + 0.1 * np.random.randn(200, 1)
    
    # Normalizar
    X_norm = (X - X.mean()) / X.std()
    
    funcoes = ['sigmoid', 'tanh', 'relu', 'leaky_relu']
    
    plt.figure(figsize=(15, 10))
    
    for i, funcao in enumerate(funcoes):
        print(f"\nTestando {funcao.upper()} para regressão...")
        
        try:
            # Ajustar parâmetros para cada função
            if funcao in ['relu', 'leaky_relu']:
                taxa = 0.001
                epocas = 5000
            else:
                taxa = 0.01
                epocas = 3000
            
            mlp = MLPComFuncoesAtivacao(camadas=[1, 20, 15, 1], 
                                      taxa_aprendizado=taxa, 
                                      funcao_ativacao=funcao)
            
            historico_erro = mlp.treinar(X_norm, y, epocas=epocas, verbose=False)
            
            # Fazer previsões
            previsoes = mlp.prever(X_norm)
            mse = np.mean((previsoes - y) ** 2)
            
            print(f"MSE final: {mse:.4f}")
            
            # Plotar resultados
            plt.subplot(2, 2, i+1)
            plt.scatter(X, y, alpha=0.5, label='Dados reais', s=10)
            
            # Ordenar para plotar linha suave
            indices = np.argsort(X.ravel())
            plt.plot(X[indices], previsoes[indices], 'r-', linewidth=2, 
                    label=f'Previsões {funcao.upper()}')
            
            plt.title(f'{funcao.upper()}: MSE = {mse:.4f}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Erro com {funcao}: {e}")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=== Demonstração de Funções de Ativação ===")
    
    # Visualizar as funções
    print("\n1. Visualizando funções de ativação e suas derivadas...")
    visualizar_funcoes_ativacao()
    
    # Comparar no problema XOR
    print("\n2. Comparando funções no problema XOR...")
    comparar_funcoes_ativacao_xor()
    
    # Testar em regressão
    print("\n3. Testando funções em problema de regressão...")
    exemplo_regressao_com_diferentes_ativacoes()
    
    print("\n=== Demonstração Concluída ===")