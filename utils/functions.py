import numpy as np
from math import atan2, acos, cos, sin

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
    #c3 = np.clip(c3, -1, 1)

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

def mapping(theta_tgt):
    dtheta = np.array([1, 1, -1, 1, 1, 1]) * (theta_tgt - np.radians([0, 90, 0, 0, 0, 0]))
    theta_out = np.rad2deg(dtheta) + [80, 80, 50, 50, 0, 0]
    return theta_out

def max_range(z:int):
    if(z < 0.284):
        #ARMLEN_SQD = 0.080656 # normal = 0.284
        return np.sqrt(0.080656 - z**2)
    else:
        print('z maior do que braço')