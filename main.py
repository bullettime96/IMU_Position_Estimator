import csv
from scipy.signal import butter, lfilter, freqz

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IMUSensor import IMUSensor as IMU

ROW_FORMAT = ['Timestamp', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
SAMPLING_FREQ = 1 / 0.003
CUTOFF_FREQ = 25
ORDER = 6

with open('output.csv', newline="\r\n") as csvfile:
    reader = csv.reader(csvfile, dialect='unix')
    sensor = IMU(rawData=reader, rawDataFormat=ROW_FORMAT)
    print(sensor)
    sensor.parse()
    print(sensor)
    sensor.calculateAcceleration()
    print(sensor)
    sensor.calculateVelocity()
    print(sensor)
    sensor.calculateTrajectory()
    print(sensor)

    sensor.plotRawMeasurements()
    sensor.plotCalculation()
    sensor.plotComparison()
    sensor.plotFiltering()
    # plotAnimatedAccelerationEstimated(sensor, filtered=True)
    # plotAnimatedAccelerationCompensated(sensor, filtered=True)
    sensor.plotAccelerationAndVelocity()
    sensor.plotAccelerationAndVelocityFiltered()
