import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt

class Camera:
    def __init__(self, webcam_index=0):
        self.webcam_index = webcam_index
        self.cap = cv2.VideoCapture(webcam_index)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.parameters = cv2.aruco.DetectorParameters()
        
        if not self.cap.isOpened():
            raise ValueError(f"Não foi possível acessar a câmera com o índice {webcam_index}")
        
    def get_aruco0_positions(self, plot_image=False, return_base64=False):
        self.check_camera()
        ret, frame = self.cap.read()
        
        if not ret:
            print("Erro ao capturar a imagem da câmera.")
            return None, None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if ids is not None and 0 in ids:
            idx = list(ids).index(0)
            corner = corners[idx]
            x = int(corner[0][0][0])
            y = int(corner[0][0][1])

            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        else:
            frame = cv2.aruco.drawDetectedMarkers(frame, [], np.array([]))
            x, y = None, None

        if plot_image:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title("Imagem Capturada")
            plt.show()

        if return_base64:
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return x, y, img_base64

        return x, y, None
    
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
