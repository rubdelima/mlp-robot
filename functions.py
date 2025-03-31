import numpy as np
from math import atan2, acos, cos, sin
from sklearn.preprocessing import MinMaxScaler

def ikine(x_target, l1, l2, l3):
    """
    Calcula a configuração das juntas para um robô com 4 eixos, dado um ponto alvo no espaço.

    Parâmetros:
    x_target (list): Posição final desejada do efetuador [x, y, z].
    l1 (float): Comprimento do primeiro link do braço.
    l2 (float): Comprimento do segundo link do braço.
    l3 (float): Comprimento do terceiro link do braço.

    Retorno:
    theta_target (list): Configuração das juntas [theta1, theta2, theta3, theta4].
    """
    # Calcula o ângulo phi, baseado nas coordenadas x e z
    phi = atan2(x_target[2], abs(x_target[0]))

    # Ajuste do ângulo phi (10 graus conforme você mencionou)
    phi = np.deg2rad(10)

    cphi = cos(phi)
    sphi = sin(phi)

    # Cálculo de theta1
    theta1 = atan2(x_target[1], x_target[0])

    # Cálculo do novo x, considerando a posição
    new_x = np.sqrt(x_target[0]**2 + x_target[1]**2)

    # Calculando x2 e z2
    x2 = new_x - l3 * abs(cphi)
    z2 = x_target[2] - l3 * sphi

    # Cálculo do c3 (coseno de theta3)
    c3 = (x2**2 + z2**2 - l1**2 - l2**2) / (2 * l1 * l2)

    # Limitar c3 para o intervalo [-1, 1] para evitar erros no acos
    c3 = np.clip(c3, -1, 1)

    # Calculando theta3
    theta3 = -acos(c3)

    # Cálculo de s3 (seno de theta3)
    s3 = sin(theta3)

    # Cálculo de theta2
    c2 = ((l1 + l2 * c3) * x2 + l2 * s3 * z2) / (x2**2 + z2**2)
    s2 = ((l1 + l2 * c3) * z2 - l2 * s3 * x2) / (x2**2 + z2**2)

    theta2 = atan2(s2, c2)

    # Cálculo de theta4
    theta4 = phi - (theta2 + theta3)

    # Retorna os valores das juntas
    theta_target = [theta1, theta2, theta3, theta4, 0, 0]
    return theta_target

def fkine(theta, l1, l2, l3, z):
    """
    Calcula a posição do efetuador final a partir das configurações das juntas.

    Parâmetros:
    theta (list): Configuração das juntas [theta1, theta2, theta3, theta4].
    l1 (float): Comprimento do primeiro link do braço.
    l2 (float): Comprimento do segundo link do braço.
    l3 (float): Comprimento do terceiro link do braço.
    z (float): A altura do efetuador final (eixo z).

    Retorno:
    pos (list): Posição cartesiana do efetuador [x, y, z].
    """
    # Extrai os ângulos das juntas
    theta1 = theta[0]
    theta2 = theta[1]
    theta3 = theta[2]
    theta4 = theta[3]
    
    # Cálculo da posição do efetuador final
    x = cos(theta1) * (l1 * cos(theta2) + l2 * cos(theta2 + theta3) + l3 * cos(theta2 + theta3 + theta4))
    y = sin(theta1) * (l1 * cos(theta2) + l2 * cos(theta2 +  theta3) + l3 * cos(theta2 + theta3 + theta4))

    # Retorna a posição cartesiana [x, y, z]
    pos = [x, y, z]
    return pos

def gen_data_train(num_amostras=1000, l1=0.1, l2=0.124, l3=0.06, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5), 
                   z_range=(-0.5, 0.5), normalize=False, folds=0):
    """
    Gera dados de treinamento para um robô com 4 eixos.

    Parâmetros:
    num_amostras (int): Número de amostras a serem geradas.
    l1 (float): Comprimento do primeiro link do braço.
    l2 (float): Comprimento do segundo link do braço.
    l3 (float): Comprimento do terceiro link do braço.
    x_range (tuple): Intervalo para as coordenadas x da posição final.
    y_range (tuple): Intervalo para as coordenadas y da posição final.
    z_range (tuple): Intervalo para as coordenadas z da posição final.
    normalize (bool): Se True, normaliza as entradas e saídas.
    folds (int): Número de folds para validação cruzada (não utilizado na geração de dados, apenas para a estrutura).

    Retorno:
    dados_entradas (np.array): Dados de entrada gerados.
    dados_saidas (np.array): Dados de saída gerados.
    """
    dados_entradas = []  # Coordenadas (x, y, z), posição final e últimas juntas
    dados_saidas = []    # Configurações finais das juntas (axis0, axis1, axis2, axis3)

    for _ in range(num_amostras):
        # Gerando coordenadas aleatórias dentro de um espaço de trabalho
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        z = np.random.uniform(z_range[0], z_range[1])

        # Gerando a posição final aleatória
        x_final = np.random.uniform(x_range[0], x_range[1])
        y_final = np.random.uniform(y_range[0], y_range[1])
        z_final = np.random.uniform(z_range[0], z_range[1])

        # Chamando a função de cinemática inversa para calcular as juntas finais
        theta_final = ikine([x_final, y_final, z_final], l1, l2, l3)

        # Armazenando as entradas (posição inicial, posição final e configurações de juntas)
        dados_entradas.append([x, y, z, x_final, y_final, z_final] + theta_final[:4])  # Considera 4 eixos
        dados_saidas.append(theta_final[:4])  # Configuração final das juntas (axis0, axis1, axis2, axis3)

    # Normalização (se necessário)
    if normalize:
        scaler_entradas = MinMaxScaler()
        dados_entradas = scaler_entradas.fit_transform(dados_entradas)

        scaler_saidas = MinMaxScaler()
        dados_saidas = scaler_saidas.fit_transform(dados_saidas)

    return np.array(dados_entradas), np.array(dados_saidas)
