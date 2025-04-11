import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt
from utils.functions import *

class Camera:
    def __init__(self, camera_height, webcam_index=0):
        self.webcam_index = webcam_index
        self.cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.camera_height = camera_height
        self.x_start, self.y_start = 0, 0
        if not self.cap.isOpened():
            raise ValueError(f"Não foi possível acessar a câmera com o índice {webcam_index}")
        
    def get_aruco_positions(self, aruco_value=0, real_pos=True, plot_image=False, return_base64=False, focalLength=1500):
        self.check_camera()
        ret, frame = self.cap.read()
        
        #if(self.x_start is None):
        #    raise TypeError("Os valores de x_start e y_start ainda não foram registrados")

        if not ret:
            print("Erro ao capturar a imagem da câmera.")
            return None, None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        x0, y0, x1, y1, x2, y2, x3, y3, xc, yc, diagonal, img_base64 = [None]*12

        if ids is not None and aruco_value in ids:
            idx = list(ids).index(aruco_value)
            corner = corners[idx][0]

            x0, y0 = int(corner[0][0]), int(corner[0][1])
            x1, y1 = int(corner[1][0]), int(corner[1][1])
            x2, y2 = int(corner[2][0]), int(corner[2][1])
            x3, y3 = int(corner[3][0]), int(corner[3][1])
            
            if(x0 is not None):
                xc_px = int((x0 + x2) / 2) - self.x_start # X_central
                yc_px = int((y0 + y2) / 2) - self.y_start # Y_central

            diagonal = np.sqrt((x0 - x2)**2 + (y0 - y2)**2)

            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        else:
            frame = cv2.aruco.drawDetectedMarkers(frame, [], np.array([]))

        if plot_image:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title("Imagem Capturada")
            plt.show()

        if return_base64:
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        if(x0 is None):
            return None, None, None, None, None

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        #return x0, y0, x1, y1, x2, y2, x3, y3, xc, yc, diagonal, img_base64, width, height

        if(real_pos):
            xc, yc = img2real((xc_px, yc_px), self.camera_height, focalLength)
            return xc_px, yc_px, xc, yc, diagonal
        return xc_px, yc_px, None, None, diagonal
    
    def show_arucos(self):
        self.check_camera()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Erro ao capturar a imagem da câmera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

            if ids is not None:
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            cv2.imshow("ArUco Detectado - Pressione 'q' para sair", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def set_origin(self):
        #x, y, _, _, _ = self.get_aruco_positions(real_pos=False, plot_image=True)
        self.check_camera()
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        print(ids)
        x0, y0, x1, y1, x2, y2, x3, y3, xc, yc, diagonal, img_base64 = [None]*12

        if ids is not None and corners is not None:
            idx = list(ids).index(0)
            corner = corners[idx][0]

            x0, y0 = int(corner[0][0]), int(corner[0][1])
            x1, y1 = int(corner[1][0]), int(corner[1][1])
            x2, y2 = int(corner[2][0]), int(corner[2][1])
            x3, y3 = int(corner[3][0]), int(corner[3][1])
            
            xc_px = int((x0 + x2) / 2) # X_central
            yc_px = int((y0 + y2) / 2) # Y_central

            self.x_start, self.y_start = xc_px, yc_px
            print(f'Ponto inicial atualizado: {self.x_start}, {self.y_start}')
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title("Imagem Capturada")
            plt.show()
        else:
            print('Erro ao capturar o Aruco. Tente novamente.')
            

    def release(self):
        self.cap.release()
    
    def check_camera(self):
        if not self.cap.isOpened():
            try:
                self.renitialize()
            except Exception as e:
                print(f"Não foi possível reinicializar a câmera com o índice {self.webcam_index}. Erro: {e}")
                raise e
    
    def renitialize(self):
        self.cap = cv2.VideoCapture(self.webcam_index)
        
        if not self.cap.isOpened():
            raise ValueError(f"Não foi possível reinicializar a câmera com o índice {self.webcam_index}")

    #def dist_and_rel_pos(x_px, y_px): # mudar o nome
    #    if(x_px is not None):
    #        real_pos = img2real((x_px, y_px))
            

'''% Cálculo de distância e posição relativa
                if ~isempty(markers{1})
                    aruco_detected = true;
                    pos0 = markers{1};
                    pos0 = pos0(1,:);
                    pos = img2real(pos0, obj.cameraHeight);
                    real_pos  =pos;
                    positions_obj = markers{2};
                    for i = 1:size(markers{2},1)
                        obj_positions = [obj_positions; img2real(positions_obj(i,:), obj.cameraHeight)]; 
                    end
                    
                    
                    % Exibe informações
                    textPos = pos0;
                    imgMarked = insertText(imgMarked, textPos + [0, 30], sprintf('X: %.2f m, Y: %.2f m', pos(1), pos(2)), 'FontSize', 18, 'BoxColor', 'blue');
                end'''