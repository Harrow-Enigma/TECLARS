import time
import serial
import numpy as np

def decode(data):
    return [i for i in data]

def norm(data):
    dat = np.asarray(data, dtype=np.float32)
    dat = (dat - 127.5) / 127.5
    return dat

class Interface():
    def __init__(self, port, br=115200):
        self.ser = serial.Serial(port, baudrate=br)
        self.ser.reset_input_buffer()
    
    def exchange(self, text, color):
        self.ser.write((text + '\n').encode('ascii'))
        time.sleep(0.001)
        self.ser.write((str(color)).encode('ascii'))

        sensors = self.ser.readline()
        sensors = sensors.decode('ascii') #.split(',')
        # sensors = tuple(map(lambda x:float(x), sensors))

        return int(sensors)

    def close(self):
        self.ser.close()
