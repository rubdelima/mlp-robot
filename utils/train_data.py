from robot.model import Robot, FakeRobot
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
import json
import traceback
from IPython.display import clear_output, display
from utils.camera import Camera
import numpy as np
from utils.functions import *
from sklearn.preprocessing import MinMaxScaler
import time
from random import choice

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
                x0, y0, x1, y1, x2, y2, x3, y3, xc, yc, diagonal, b64, width, height = camera.get_aruco0_positions(plot_image=show_image, return_base64=capture_image) 
                
                data[data_key] = [axis0, axis1, axis2,axis3, x0, y0, x1, y1, x2, y2, x3, y3, xc, yc, diagonal, width, height, b64]
                
                if capture_image:
                    data[data_key].append(b64)
                
                last_position = (axis0, axis1, axis2, axis3)
            
    
    except Exception as e:
        print(f"Erro ao testar a posição: {e}")
        print(traceback.format_exception(e))
    
    finally:
        save_results(data, collection_name)
        df = pd.DataFrame(data.values(), columns=["axis0", "axis1", "axis2", "axis3", "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3", "x_central", "y_central", 'diagonal', 'width', 'height'] + (["b64_image"] if capture_image else []))
        df.to_csv(f"./data/{collection_name}.csv", index=False)
        
        try:
            camera.release()
        except:
            print("Houve um erro ao fechar a câmera")
        
        return df

def new_data_train(
	    robot : Robot | FakeRobot, # Robô que iremos controlar
	    camera : Camera, # Camera que iremos usar para capturar o ArUco
	    z=0.12,
        x_px_size: int = 640,
        y_px_size: int = 480,
        tam_diagonal:float=0.06, # em m
        num_samples = 10,
        capture_image : bool = False, # Para salvar como um dos dados a imagem em b64
        show_image : bool = False # Para visualzar a imagem em tempo real
	    )->pd.DataFrame:
    
    '''axis_positions = list(product(
        range(*axis0_range, step),range(*axis1_range, step),
        range(*axis2_range, step),range(*axis3_range, step)
    ))'''
    
    max_axis_pos = max_range(z) # aqui "axis" significa eixo do eixo cartesiano mesmo

    x_range = range(0.04, 2, ) # vai ter q criar um range dos x e y possíveis em m (na real cm mas blz)

    x0, y0, x1, y1, x2, y2, x3, y3, xc, yc, diagonal, b64, width, height = camera.get_aruco0_positions(plot_image=False, return_base64=capture_image)

    if(x0 is None):
        print('Aruco não localizado')
        pass

    dim_convert = lambda x: x * (tam_diagonal/diagonal)

    data = pd.DataFrame(columns=['theta0', 'theta1', 'theta2', 'theta3', 'x_camera', 'y_camera', 'x', 'y'])

    print(f'original: ({x0}, {y0}, {z})')

    #collection_name = f"0 {axis0_range}, 1 {axis1_range}, 2 {axis2_range}, 3 {axis3_range}, S {step}" + ("-b64" if capture_image else "")
    
    #data = load_results(collection_name)
    
    #tested_axis = list(map(lambda x: tuple(map(int, x.split(", "))), data.keys()))
    #axis_positions = list(filter(lambda x : x not in tested_axis, axis_positions))
    
    #robot.move_to(*axis_positions[0])
    #time.sleep(5) # Espera 5 segundos para o braço se mover para a posição desejada
    #last_position = axis_positions[0]
    
    '''i = 0
    total_positions = len(axis_positions)
    t = tqdm(axis_positions, desc=f"Testando um total de {len(axis_positions)} diferentes", total=len(axis_positions))
    '''
    try:
        i = 0
        while(i<num_samples):

            #x_px = randint(0.04,x_range)
            #y_px = rand(0.04,y_range)
            new_pos = (dim_convert(x_px), dim_convert(y_px), z)
            print(f'px: ({x_px}, {y_px}, {z})\tcm: {new_pos}')
            theta = mapping(ikine(new_pos, l1=0.1, l2=0.124, l3=0.06))

            print(f'''angulação: 
theta0 = {theta[0]} | theta1 = {theta[1]} | theta2 = {theta[2]} | theta3 = {theta[3]}''')

            #relative_pos = (new_pos[0]-x0, new_pos[1]-y0, z)

            robot.move_to(theta[0], theta[1], theta[2], theta[3])
            time.sleep(7) # Espera 5 segundos para o braço se mover para a posição desejada

            _ = camera.get_aruco0_positions(plot_image=False, return_base64=False) # teste

            #_new
            x0_new, y0_new, _, _, _, _, _, _, _, _, _, _, _, _ = camera.get_aruco0_positions(plot_image=show_image, return_base64=capture_image)

            if(x0_new is not None):

                relative_pos = (dim_convert(x0_new)-x0, dim_convert(y0_new)-y0, z)

                print(f'''posição na câmera: ({x0_new}, {y0_new}, {z})
    posição relativa à origem: {relative_pos}''')

                data.loc[i] = [theta[0], theta[1], theta[2], theta[3], x0_new, y0_new, relative_pos[0], relative_pos[1]]

                i += 1

            else:
                print('Aruco não localizado, tentando novamente')
        '''for axis0, axis1, axis2, axis3 in t:
            
            data_key = f"{axis0}, {axis1}, {axis2}, {axis3}"
            
            if data_key not in data.keys():
                
                robot.move_to(axis0, axis1, axis2, axis3)
                
                dynamic_sleep(last_position, (axis0, axis1, axis2, axis3))
                clear_output(wait=True)
                
                t.refresh()
                
                print(f"{i}/{total_positions}")
                i += 1
                x0, y0, x1, y1, x2, y2, x3, y3, xc, yc, diagonal, b64, width, height = camera.get_aruco0_positions(plot_image=show_image, return_base64=capture_image) 
                
                data[data_key] = [axis0, axis1, axis2, x0, y0, x1, y1, x2, y2, x3, y3, xc, yc, diagonal, width, height, b64]
                
                if capture_image:
                    data[data_key].append(b64)
                
                last_position = (axis0, axis1, axis2, axis3)'''
            
    
    except Exception as e:
        print(f"Erro ao testar a posição: {e}")
        camera.release()
        print(traceback.format_exception(e))
    
    finally:
        collection_name = 'teste'
        #save_results(data, collection_name)
        #df = pd.DataFrame(data.values(), columns=["axis0", "axis1", "axis2", "axis3", "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3", "x_central", "y_central", 'diagonal', 'width', 'height'] + (["b64_image"] if capture_image else []))
        data.to_csv(f"./data/new{collection_name}.csv", index=False)
        
        try:
            camera.release()
        except:
            print("Houve um erro ao fechar a câmera")
        
        return data

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
        '''x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        z = np.random.uniform(z_range[0], z_range[1])'''

        # Gerando ângulos iniciais aleatórios
        theta1 = np.random.uniform(0, 180)
        theta2 = np.random.uniform(0, 180)
        theta3 = np.random.uniform(0, 180)
        theta4 = np.random.uniform(0, 180)

        theta_start = (theta1, theta2, theta3, theta4)

        x, y, z = fkine(theta_start, l1, l2, l3, z=0.12)

        # Gerando a posição final aleatória
        x_final = np.random.uniform(x_range[0], x_range[1])
        y_final = np.random.uniform(y_range[0], y_range[1])
        z_final = np.random.uniform(z_range[0], z_range[1])

        # Gerando ângulos iniciais aleatórios
        theta1_final = np.random.uniform(0, 180)
        theta2_final = np.random.uniform(0, 180)
        theta3_final = np.random.uniform(0, 180)
        theta4_final = np.random.uniform(0, 180)

        theta_final = (theta1_final, theta2_final, theta3_final, theta4_final)

        # Chamando a função de cinemática inversa para calcular as juntas finais
        #theta_final = ikine([x_final, y_final, z_final], l1, l2, l3)

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

def get_data_train_inike(
    robot : Robot,
    camera : Camera,
    x_range : tuple[float, float],
    y_range : tuple[float, float],
    z : float = 0.12,
    step : int=10,
    l1 : float=0.1, l2 : float=0.124, l3: float = 0.06
):
    
    tested_positions = list(product(
        range(int(x_range[0]*100), int(x_range[1]*100), step),
        range(int(y_range[0]*100), int(y_range[1]*100), step)
    ))
    
    data = load_results(f"ikine{x_range, y_range, z, step}")
    
    #max_axis = max_range(z)

    last_position = (robot.axis0, robot.axis1, robot.axis2, robot.axis3)
    
    try:
        for int_x, int_y in tested_positions:
            x, y = int_x/100, int_y/100
            data_key = f"{int_x}_{int_y}_{z}"
            
            if data_key not in data:
                t0, t1, t2, t3, _, _ = ikine([x, y, z], l1, l2, l3)

                robot.move_to(t0, t1, t2, t3)
                #dynamic_sleep(last_position, (t0,t1, t2, t3))
                clear_output(wait=True)
                print("Testando posções para: ", x, y)
                time.sleep(5)
                x0, y0, x1, y1, x2, y2, x3, y3, xc, yc, diagonal, b64, width, height = camera.get_aruco0_positions(plot_image=True) 

                data[data_key] = [x, y, z, t0,t1, t2, t3, x0, y0, x1, y1, x2, y2, x3, y3, xc, yc, diagonal, b64, width, height]

                last_position = (t0, t1, t2, t3)
    except Exception as e:
        print(f"Erro ao testar a posição: {e}")
        print(traceback.format_exception(e))
    
    finally:
        save_results(data,f"ikine{(x_range, y_range, z, step)}" )
        df = pd.DataFrame(data.values(),columns=[
            "x", "y", "z", "t0", "t1", "t2", "t3",
            "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3",
            "xc", "yc", "diagonal", "b64_image", "width", "height"
        ])
        
        df.to_csv(f"./data/ikine{(x_range, y_range, z, step)}.csv", index=False)
        