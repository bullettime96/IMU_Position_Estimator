import csv
import numpy as np
import matplotlib.pyplot as plt
from IMUSensor import IMUSensor as IMU

ROW_FORMAT = ['Timestamp', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']


with open('output.csv', newline="\r\n") as csvfile:
    reader = csv.reader(csvfile, dialect='unix')
    sensor = IMU(rawData=reader, rawDataFormat=ROW_FORMAT)
    print(sensor)
    sensor.parse()
    print(sensor)
    print(sensor.getAcceleratorX())

    times = sensor.getTimestamps()
    accX = sensor.getAcceleratorX()
    accY = sensor.getAcceleratorY()
    accZ = sensor.getAcceleratorZ()
    gyroX = sensor.getGyroX()
    gyroY = sensor.getGyroY()
    gyroZ = sensor.getGyroZ()

    plt.subplot(2, 2, 1)
    plt.title('Acceleration')
    plt.plot(times, accX, times, accY, times, accZ)
    plt.subplot(2, 2, 2)
    plt.title('Acceleration in X axis only')
    plt.plot(times, accX)
    plt.subplot(2, 2, 3)
    plt.title('Rotation rate')
    plt.plot(times, gyroX, times, gyroY, times, gyroZ)
    plt.subplot(2, 2, 4)
    plt.title('Rotation rate in X axis')
    plt.plot(times, gyroX)
    plt.show()
