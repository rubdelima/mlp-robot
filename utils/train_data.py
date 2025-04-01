from robot.model import Robot, FakeRobot
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
import json
import traceback
from IPython.display import clear_output, display
from utils.camera import Camera
import numpy as np
from utils.functions import ikine
from sklearn.preprocessing import MinMaxScaler
import time

def load_results(file_name):
    try:
        with open(f'data/{file_name}.json') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_results(data: dict, file_name: str):
    with open(f'data/{file_name}.json', 'w') as f:
        json.dump(data, f, indent=4)


def dynamic_sleep(last_position, new_position, max_distance=180, min_time=1, max_time=5):
    distance = sum(abs(last - new) for last, new in zip(last_position, new_position))
    
    normalized_time = (distance / (4 * max_distance)) * (max_time - min_time) + min_time 
    
    print(f"Aguardando por {normalized_time:.2f} segundos...")
    time.sleep(normalized_time)


def real_data_train(
	    robot : Robot | FakeRobot, # Robô que iremos controlar
	    camera : Camera, # Camera que iremos usar para capturar o ArUco
	    axis0_range : tuple[int, int] = (0,180), # Range de quanto vamos variar o eixo 0 do robô
	    axis1_range : tuple[int, int] = (0,180), # Range de quanto vamos variar o eixo 1 do robô
	    axis2_range : tuple[int, int] = (0,180), # Range de quanto vamos variar o eixo 2 do robô
	    axis3_range : tuple[int, int] = (0,180), # Range de quanto vamos variar o eixo 3 do robô
        step : int = 5,  # Steps de quanto vamos variar os eixos a cada iteração
        capture_image : bool = False, # Para salvar como um dos dados a imagem em b64
        show_image : bool = False # Para visualzar a imagem em tempo real
	    )->pd.DataFrame:
    
    axis_positions = list(product(
        range(*axis0_range, step),range(*axis1_range, step),
        range(*axis2_range, step),range(*axis3_range, step)
    ))
    
    collection_name = f"0 {axis0_range}, 1 {axis1_range}, 2 {axis2_range}, 3 {axis3_range}, S {step}" + ("-b64" if capture_image else "")
    
    data = load_results(collection_name)
    
    tested_axis = list(map(lambda x: tuple(map(int, x.split(", "))), data.keys()))
    axis_positions = list(filter(lambda x : x not in tested_axis, axis_positions))
    
    robot.move_to(*axis_positions[0])
    time.sleep(5) # Espera 5 segundos para o braço se mover para a posição desejada
    last_position = axis_positions[0]
    
    i = 0
    total_positions = len(axis_positions)
    t = tqdm(axis_positions, desc=f"Testando um total de {len(axis_positions)} diferentes", total=len(axis_positions))
    
    try:
        for axis0, axis1, axis2, axis3 in t:
            
            data_key = f"{axis0}, {axis1}, {axis2}, {axis3}"
            
            if data_key not in data.keys():
                
                robot.move_to(axis0, axis1, axis2, axis3)
                
                dynamic_sleep(last_position, (axis0, axis1, axis2, axis3))
                clear_output(wait=True)
                
                t.refresh()
                
                print(f"{i}/{total_positions}")
                i += 1
                x, y , b64 = camera.get_aruco0_positions(plot_image=show_image, return_base64=capture_image)
                
                data[data_key] = [axis0, axis1, axis2, axis3, x, y]
                
                if capture_image:
                    data[data_key].append(b64)
                
                last_position = (axis0, axis1, axis2, axis3)
            
    
    except Exception as e:
        print(f"Erro ao testar a posição: {e}")
        print(traceback.format_exception(e))
    
    finally:
        save_results(data, collection_name)
        df = pd.DataFrame(data.values(), columns=["axis0", "axis1", "axis2", "axis3", "x_pos", "y_pos"] + (["b64_image"] if capture_image else []))
        df.to_csv(f"./data/{collection_name}.csv", index=False)
        
        try:
            camera.release()
        except:
            print("Houve um erro ao fechar a câmera")
        
        return df


def sim_data_train(num_amostras=1000, l1=0.1, l2=0.124, l3=0.06, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5), 
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