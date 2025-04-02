import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt

class Camera:
    def __init__(self, webcam_index=0):
        self.webcam_index = webcam_index
        self.cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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

        x0, y0, x1, y1, x2, y2, x3, y3, xc, yc, diagonal, img_base64 = [None]*12

        if ids is not None and 0 in ids:
            idx = list(ids).index(0)
            corner = corners[idx][0]

            x0, y0 = int(corner[0][0]), int(corner[0][1])
            x1, y1 = int(corner[1][0]), int(corner[1][1])
            x2, y2 = int(corner[2][0]), int(corner[2][1])
            x3, y3 = int(corner[3][0]), int(corner[3][1])

            xc = int((x0 + x2) / 2) # X_central
            yc = int((y0 + y2) / 2) # Y_central

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
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return x0, y0, x1, y1, x2, y2, x3, y3, xc, yc, diagonal, img_base64, width, height
    
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
