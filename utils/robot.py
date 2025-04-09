from serial import Serial
import numpy as np
import time
import pickle
from utils.camera import Camera

min_max_dict = {"xc_px": [-282.0, 293.0], "yc_px": [-53.0, 282.0], "xc": [-5.274, 5.076], "yc": [-0.954, 5.076]}

def reverse_min_max(value, column_name):
    min_value, max_value = min_max_dict[column_name]
    return value * (max_value - min_value) + min_value

class Robot():
    def __init__(self, serial_port, model_path='./models/mlp_base.plk'):
        self.serial = Serial(serial_port, 115200)
        time.sleep(2)
        self.reset()
        self.move()
        with open(model_path, 'rb') as f:
            self.controller = pickle.load(f)
    
    def move(self):
        
        comando = f"0:{self.axis0}, 1:{self.axis1}, 2:{self.axis2}, 3:{self.axis3}"
        self.serial.write(comando.encode())
        print(f"Comando enviado: {comando}")
        time.sleep(0.1)
        
        # Lê a resposta do Arduino
        resposta = self.serial.readline().decode().strip()
        time.sleep(0.1)
    
    def reset(self):
        self.axis0 = 80
        self.axis1 = 161.55
        self.axis2 = 171.62
        self.axis3 = 10
        self.move()
    
    def move_to(self, axis0 =None, axis1=None, axis2=None, axis3=None):
        self.axis0 = axis0 if axis0 is not None else self.axis0
        self.axis1 = axis1 if axis1 is not None else self.axis1
        self.axis2 = axis2 if axis2 is not None else self.axis2
        self.axis3 = axis3 if axis3 is not None else self.axis3
        self.move()
    
    def move_to_ikine(self, theta_target:np.ndarray):
        dtheta = np.array([1, 1, -1, 1, 1, 1]) * (theta_target - np.radians([0, 90, 0, 0, 0, 0]))
        return np.rad2deg(dtheta) + np.array((80,80,50,50,0,0))

    def rotate(self, axis0 =None, axis1=None, axis2=None, axis3=None):
        self.axis0 += (axis0 if axis0 is not None else 0)
        self.axis1 += (axis1 if axis1 is not None else 0)
        self.axis2 += (axis2 if axis2 is not None else 0)
        self.axis3 += (axis3 if axis3 is not None else 0)
        self.move()
    
    def close(self):
        self.serial.close()
    
    def show_positions(self):
        print(f"Posições: axis0={self.axis0}, axis1={self.axis1}, axis2={self.axis2}, axis3={self.axis3}")

    def get_positions(self):
        return (self.axis0, self.axis1, self.axis2, self.axis3)
    
    def go_to_display_position(self, camera:Camera):
        camera.get_aruco0_positions()
        camera.get_aruco0_positions()
        xc_px, yc_px, xc, yc, diagonal = camera.get_aruco0_positions(plot_image=True)
        xc_px, yc_px, xc, yc = reverse_min_max(xc_px, 'xc_px'), \
                reverse_min_max(yc_px, 'yc_px'), reverse_min_max(xc, 'xc'), \
                reverse_min_max(yc, 'yc')
                
        predicted_angles = self.controller.predict([[xc_px, yc_px, xc, yc]])
        t0, t1, t2, t3 = predicted_angles[0]
        print(f"Posições Preditas: axis0={t0}, axis1={t1}, axis2={t2}, axis3={t3}")
        # self.move_to(t0, t1, t2, t3)
    