from serial import Serial
import time

class Robot():
    def __init__(self, serial_port):
        self.serial = Serial(serial_port, 115200)
        time.sleep(2)
        self.reset()
        self.move()
    
    def move(self):
        
        comando = f"0:{self.axis0%180}, 1:{self.axis1%180}, 2:{self.axis2%180}, 3:{self.axis3%180}"
        self.serial.write(comando.encode())
        print(f"Comando enviado: {comando}")
        time.sleep(0.1)
        
        # Lê a resposta do Arduino
        resposta = self.serial.readline().decode().strip()
        time.sleep(0.1)
    
    def reset(self):
        self.axis0 = 80
        self.axis1 = 75
        self.axis2 = 50
        self.axis3 = 0
        self.move()
    
    def move_to(self, axis0 =None, axis1=None, axis2=None, axis3=None):
        self.axis0 = axis0 if axis0 is not None else self.axis0
        self.axis1 = axis1 if axis1 is not None else self.axis1
        self.axis2 = axis2 if axis2 is not None else self.axis2
        self.axis3 = axis3 if axis3 is not None else self.axis3
        self.move()
    
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