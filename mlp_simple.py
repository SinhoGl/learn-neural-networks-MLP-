import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class MLPSimples:
    """
    Implementação simples de uma rede neural MLP (Multi-Layer Perceptron)
    para fins educacionais.
    """
    
    def __init__(self, camadas: List[int], taxa_aprendizado: float = 0.01):
        """
        Inicializa a rede MLP.
        
        Args:
            camadas: Lista com o número de neurônios em cada camada
                    Ex: [2, 4, 3, 1] = entrada(2), oculta1(4), oculta2(3), saída(1)
            taxa_aprendizado: Taxa de aprendizado para o treinamento
        """
        self.camadas = camadas
        self.taxa_aprendizado = taxa_aprendizado
        self.num_camadas = len(camadas)
        
        # Inicializar pesos e bias aleatoriamente
        self.pesos = []
        self.bias = []
        
        for i in range(self.num_camadas - 1):
            # Inicialização Xavier/Glorot
            peso = np.random.randn(camadas[i], camadas[i+1]) * np.sqrt(2.0 / camadas[i])
            bias = np.zeros((1, camadas[i+1]))
            
            self.pesos.append(peso)
            self.bias.append(bias)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Função de ativação sigmoid"""
        # Clipping para evitar overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def derivada_sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Derivada da função sigmoid"""
        return x * (1 - x)
    
    def forward_pass(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Executa o forward pass da rede.
        
        Args:
            X: Dados de entrada (batch_size, num_features)
            
        Returns:
            ativacoes: Lista com as ativações de cada camada
            z_valores: Lista com os valores antes da ativação
        """
        ativacoes = [X]
        z_valores = []
        
        entrada_atual = X
        
        for i in range(self.num_camadas - 1):
            # Calcular z = X * W + b
            z = np.dot(entrada_atual, self.pesos[i]) + self.bias[i]
            z_valores.append(z)
            
            # Aplicar função de ativação
            ativacao = self.sigmoid(z)
            ativacoes.append(ativacao)
            
            entrada_atual = ativacao
        
        return ativacoes, z_valores
    
    def backward_pass(self, X: np.ndarray, y: np.ndarray, ativacoes: List[np.ndarray]) -> None:
        """
        Executa o backward pass (backpropagation) e atualiza os pesos.
        
        Args:
            X: Dados de entrada
            y: Rótulos verdadeiros
            ativacoes: Ativações calculadas no forward pass
        """
        m = X.shape[0]  # Número de exemplos
        
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
                erro_atual = np.dot(erro_atual, self.pesos[i].T) * self.derivada_sigmoid(ativacoes[i])
        
        # Atualizar pesos e bias
        for i in range(self.num_camadas - 1):
            self.pesos[i] -= self.taxa_aprendizado * gradientes_pesos[i]
            self.bias[i] -= self.taxa_aprendizado * gradientes_bias[i]
    
    def treinar(self, X: np.ndarray, y: np.ndarray, epocas: int = 1000, verbose: bool = True) -> List[float]:
        """
        Treina a rede neural.
        
        Args:
            X: Dados de entrada
            y: Rótulos verdadeiros
            epocas: Número de épocas de treinamento
            verbose: Se deve imprimir o progresso
            
        Returns:
            historico_erro: Lista com o erro de cada época
        """
        historico_erro = []
        
        for epoca in range(epocas):
            # Forward pass
            ativacoes, _ = self.forward_pass(X)
            
            # Calcular erro (Mean Squared Error)
            erro = np.mean((ativacoes[-1] - y) ** 2)
            historico_erro.append(erro)
            
            # Backward pass
            self.backward_pass(X, y, ativacoes)
            
            # Imprimir progresso
            if verbose and epoca % 100 == 0:
                print(f"Época {epoca}, Erro: {erro:.6f}")
        
        return historico_erro
    
    def prever(self, X: np.ndarray) -> np.ndarray:
        """
        Faz previsões com a rede treinada.
        
        Args:
            X: Dados de entrada
            
        Returns:
            Previsões da rede
        """
        ativacoes, _ = self.forward_pass(X)
        return ativacoes[-1]
    
    def avaliar_acuracia(self, X: np.ndarray, y: np.ndarray, limiar: float = 0.5) -> float:
        """
        Avalia a acurácia da rede (para problemas de classificação binária).
        
        Args:
            X: Dados de entrada
            y: Rótulos verdadeiros
            limiar: Limiar para classificação binária
            
        Returns:
            Acurácia da rede
        """
        previsoes = self.prever(X)
        previsoes_binarias = (previsoes > limiar).astype(int)
        acuracia = np.mean(previsoes_binarias == y)
        return acuracia


def exemplo_uso():
    """
    Exemplo de uso da rede MLP com o problema XOR.
    """
    print("=== Exemplo: Problema XOR ===")
    
    # Dados do problema XOR
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    # Criar e treinar a rede
    # Arquitetura: 2 entradas -> 4 neurônios ocultos -> 1 saída
    mlp = MLPSimples(camadas=[2, 4, 1], taxa_aprendizado=0.2)
    
    print("Treinando a rede...")
    historico_erro = mlp.treinar(X, y, epocas=2000, verbose=True)
    
    # Testar a rede
    print("\n=== Resultados ===")
    previsoes = mlp.prever(X)
    
    for i in range(len(X)):
        entrada = X[i]
        esperado = y[i][0]
        previsto = previsoes[i][0]
        print(f"Entrada: {entrada} | Esperado: {esperado} | Previsto: {previsto:.4f}")
    
    # Calcular acurácia
    acuracia = mlp.avaliar_acuracia(X, y)
    print(f"\nAcurácia: {acuracia:.2%}")
    
    # Plotar curva de erro
    plt.figure(figsize=(10, 6))
    plt.plot(historico_erro)
    plt.title('Curva de Aprendizado - Problema XOR')
    plt.xlabel('Época')
    plt.ylabel('Erro (MSE)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    exemplo_uso()