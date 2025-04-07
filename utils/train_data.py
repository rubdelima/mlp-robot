from utils.robot import Robot
import pandas as pd
from itertools import product
import json
import traceback
from IPython.display import clear_output
from utils.camera import Camera
import numpy as np
from utils.functions import *
import time
from typing import Literal, Optional

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

def range_filter(values):
    x, y , z= values
    
    return (np.sqrt(x**2 + y**2) > 0.04)

def calculate_distance(p1, p2):
    # Distância Euclidiana considerando tanto as posições (x, y, z) quanto os ângulos (t0, t1, t2, t3)
    pos_diff = np.array(p1[:3]) - np.array(p2[:3])  # Diferença nas coordenadas (x, y, z)
    angles_diff = np.array(p1[3:]) - np.array(p2[3:])  # Diferença nos ângulos (t0, t1, t2, t3)
    
    return np.linalg.norm(np.concatenate((pos_diff, angles_diff)))  # Retorna a distância total

def sort_positions(ikine_positions):
    sorted_positions = [ikine_positions.pop(0)]
    
    while ikine_positions:
        last_position = sorted_positions[-1]
        closest_point = min(ikine_positions, key=lambda x: calculate_distance(last_position, x))
        sorted_positions.append(closest_point)
        ikine_positions.remove(closest_point)
    
    return sorted_positions

def generate_positions(x_range, y_range, z_range, step=10):
    return list(
        filter(
            range_filter,
            map(
                lambda x : (x[0]/1000, x[1]/1000, x[2]/1000),
                product(
                    range(int(x_range[0]*1000), int(x_range[1]*1000), step),
                    range(int(y_range[0]*1000), int(y_range[1]*1000), step), 
                    range(int(z_range[0]*1000), int(z_range[1]*1000), step),
                )
            )
        )
    )

def check_theta_range(thetas):
    out_of_range = sum([theta is not None and (theta < 0 or theta > 180) for theta in thetas])
    return out_of_range

def calculate_joint_angles(tested_positions, l1, l2, l3, return_type:Literal["list", "dataframe"]="list"):
    data = []
    
    for x, y, z in tested_positions:
        try:
            t0, t1, t2, t3, _, _ = mapping(ikine([x, y, z], l1, l2, l3))
            angles = [x, y, z, t0, t1, t2, t3]
            if return_type == "dataframe":
                out_of_range = check_theta_range([t0, t1, t2, t3])
                angles.append(out_of_range)
            data.append(angles)
        except:
            continue
    
    if return_type == "dataframe":
        return pd.DataFrame(data, columns=["x", "y", "z", "t0", "t1", "t2", "t3", "out_of_range"])
    
    return sort_positions(data)


def get_data_train_inike(
    robot : Robot,
    camera : Camera,
    x_range : tuple[float, float],
    y_range : tuple[float, float],
    z_range : tuple[float, float] = (0.12, 0.121),
    step : int=10,
    l1 : float=0.1, l2 : float=0.124, l3: float = 0.06,
    max_samples :Optional[int] =  None
):
    
    robot.move_to(80.0, 163.6495484174921, 171.03985026429626, 7.390301846804171)

    camera.set_origin()

    initial_time = time.time()
    
    tested_positions = generate_positions(x_range, y_range, z_range, step)
    
    ikine_positions = calculate_joint_angles(tested_positions, l1, l2, l3, "list")
    
    file_name = f"ikine{x_range, y_range, z_range, step}"
    data = load_results(file_name)
    
    last_position = (robot.axis0, robot.axis1, robot.axis2, robot.axis3)
    
    if max_samples is not None:
        tested_positions = tested_positions[:max_samples]
    
    total_positions = len(ikine_positions)
    
    print(f"Iniciando os testes após {time.time() - initial_time} segundos")
    
    try:
        for i , val in enumerate(ikine_positions):
            x, y,z, t0, t1, t2, t3 = val
            
            data_key = f"{x}, {y}, {z}, {t0}, {t1}, {t2}, {t3}"
            
            if data_key not in data:
                
                robot.move_to(t0, t1, t2, t3)
                print(f"Testando posição: {i} de {total_positions}")
                print(f"Movendo para: {t0}, {t1}, {t2}, {t3}")
                dynamic_sleep(last_position, (t0,t1, t2, t3))
                
                clear_output(wait=True)
                camera.get_aruco0_positions()
                camera.get_aruco0_positions()
                
                #x0, y0, x1, y1, x2, y2, x3, y3, xc, yc, diagonal, b64, width, height = camera.get_aruco0_positions(plot_image=True) 

                xc_px, yc_px, xc, yc, diagonal = camera.get_aruco0_positions(plot_image=True)

                data[data_key] = [x, y, z, t0, t1, t2, t3, xc_px, yc_px, xc, yc, diagonal]

                last_position = (t0, t1, t2, t3)
    
    except Exception as e:
        print(f"Erro ao testar a posição: {e}")
        print(traceback.format_exception(e))
    
    finally:
        save_results(data,file_name)
        df = pd.DataFrame(data.values(),columns=[
            "x", "y", "z", "t0", "t1", "t2", "t3",
            "xc_px", "yc_px", 'xc', "yc", "diagonal"
        ])
        
        df.to_csv(f"./data/{file_name}.csv", index=False)
        