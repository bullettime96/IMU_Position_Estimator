import csv
from scipy.signal import butter, lfilter, freqz

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IMUSensor import IMUSensor as IMU

ROW_FORMAT = ['Timestamp', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
SAMPLING_FREQ = 1/0.003
CUTOFF_FREQ = 25
ORDER = 6

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def update3D(iteration, RxData, RyData, RzData, RxLines, RyLines, RzLines, RLine):
    RxLines[0].set_data(np.array([0, RxData[iteration]]), np.array([0, 0]))
    RxLines[0].set_3d_properties(np.array([0, 0]))
    RxLines[1].set_data(np.array([0, RxData[iteration]]), np.array([RyData[iteration], RyData[iteration]]))
    RxLines[1].set_3d_properties(np.array([0, 0]))
    RxLines[2].set_data(np.array([0, RxData[iteration]]), np.array([0, 0]))
    RxLines[2].set_3d_properties(np.array([RzData[iteration], RzData[iteration]]))
    RxLines[3].set_data(np.array([0, RxData[iteration]]), np.array([RyData[iteration], RyData[iteration]]))
    RxLines[3].set_3d_properties(np.array([RzData[iteration], RzData[iteration]]))

    RyLines[0].set_data(np.array([0, 0]), np.array([0, RyData[iteration]]))
    RyLines[0].set_3d_properties(np.array([0, 0]))
    RyLines[1].set_data(np.array([RxData[iteration], RxData[iteration]]), np.array([0, RyData[iteration]]))
    RyLines[1].set_3d_properties(np.array([0, 0]))
    RyLines[2].set_data(np.array([0, 0]), np.array([0, RyData[iteration]]))
    RyLines[2].set_3d_properties(np.array([RzData[iteration], RzData[iteration]]))
    RyLines[3].set_data(np.array([RxData[iteration], RxData[iteration]]), np.array([0, RyData[iteration]]))
    RyLines[3].set_3d_properties(np.array([RzData[iteration], RzData[iteration]]))

    RzLines[0].set_data(np.array([0, 0]), np.array([0, 0]))
    RzLines[0].set_3d_properties(np.array([0, RzData[iteration]]))
    RzLines[1].set_data(np.array([RxData[iteration], RxData[iteration]]), np.array([0, 0]))
    RzLines[1].set_3d_properties(np.array([0, RzData[iteration]]))
    RzLines[2].set_data(np.array([0, 0]), np.array([RyData[iteration], RyData[iteration]]))
    RzLines[2].set_3d_properties(np.array([0, RzData[iteration]]))
    RzLines[3].set_data(np.array([RxData[iteration], RxData[iteration]]), np.array([RyData[iteration], RyData[iteration]]))
    RzLines[3].set_3d_properties(np.array([0, RzData[iteration]]))

    RLine.set_data(np.array([0, RxData[iteration]]), np.array([0, RyData[iteration]]))
    RLine.set_3d_properties(np.array([0, RzData[iteration]]))

def plotRawMeasurements(sensor):
    plt.subplot(4, 2, 1)
    plt.title('Acceleration in X Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRawAcceleratorX())
    plt.subplot(4, 2, 3)
    plt.title('Acceleration in Y Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRawAcceleratorY())
    plt.subplot(4, 2, 5)
    plt.title('Acceleration in Z Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRawAcceleratorZ())
    plt.subplot(4, 2, 7)
    plt.title('Total Acceleration')
    plt.plot(sensor.getTimestamps(), sensor.getTotalAcceleration())
    plt.subplot(4, 2, 2)
    plt.title('Rotation rate around X Axis')
    plt.plot(sensor.getTimestamps(), sensor.getGyroX())
    plt.subplot(4, 2, 4)
    plt.title('Rotation rate around Y Axis')
    plt.plot(sensor.getTimestamps(), sensor.getGyroY())
    plt.subplot(4, 2, 6)
    plt.title('Rotation rate around Z Axis')
    plt.plot(sensor.getTimestamps(), sensor.getGyroZ())
    plt.show()

def plotCalculation(sensor):
    plt.subplot(3, 3, 1)
    plt.title('Normalized raw acceleration in X Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRxAcc())
    plt.subplot(3, 3, 4)
    plt.title('Normalized raw acceleration in Y Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRyAcc())
    plt.subplot(3, 3, 7)
    plt.title('Normalized raw acceleration in Z Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRzAcc())
    plt.subplot(3, 3, 2)

    plt.title('Compensating acceleration from Gyro in X Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRxGyro())
    plt.subplot(3, 3, 5)
    plt.title('Compensating acceleration from Gyro in Y Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRyGyro())
    plt.subplot(3, 3, 8)
    plt.title('Compensating acceleration from Gyro in Z Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRzGyro())

    plt.subplot(3, 3, 3)
    plt.title('Normalized estimated acceleration in X Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRxEst())
    plt.subplot(3, 3, 6)
    plt.title('Normalized estimated acceleration in Y Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRyEst())
    plt.subplot(3, 3, 9)
    plt.title('Normalized estimated acceleration in Z Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRzEst())

    plt.show()

def plotComparison(sensor):
    plt.subplot(4, 3, 1)
    plt.title('Acceleration in X Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRawAcceleratorX())
    plt.subplot(4, 3, 4)
    plt.title('Acceleration in Y Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRawAcceleratorY())
    plt.subplot(4, 3, 7)
    plt.title('Acceleration in Y Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRawAcceleratorZ())
    plt.subplot(4, 3, 10)
    plt.title('Total Acceleration')
    plt.plot(sensor.getTimestamps(), sensor.getTotalAcceleration())

    plt.subplot(4, 3, 2)
    plt.title('Normalized estimated acceleration in X Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRxEst())
    plt.subplot(4, 3, 5)
    plt.title('Normalized estimated acceleration in Y Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRyEst())
    plt.subplot(4, 3, 8)
    plt.title('Normalized estimated acceleration in Z Axis')
    plt.plot(sensor.getTimestamps(), sensor.getRzEst())

    plt.subplot(4, 3, 3)
    plt.title('Compensated acceleration in X Axis')
    plt.plot(sensor.getTimestamps(), sensor.getAccXCompensated())
    plt.subplot(4, 3, 6)
    plt.title('Compensated acceleration in Y Axis')
    plt.plot(sensor.getTimestamps(), sensor.getAccYCompensated())
    plt.subplot(4, 3, 9)
    plt.title('Compensated acceleration in Z Axis')
    plt.plot(sensor.getTimestamps(), sensor.getAccZCompensated())

    plt.show()

def plotFiltering(sensor):
    plt.subplot(3, 2, 1)
    plt.title('Compensated acceleration in X Axis')
    plt.plot(sensor.getTimestamps(), sensor.getAccXCompensated())
    plt.subplot(3, 2, 3)
    plt.title('Compensated acceleration in Y Axis')
    plt.plot(sensor.getTimestamps(), sensor.getAccYCompensated())
    plt.subplot(3, 2, 5)
    plt.title('Compensated acceleration in Z Axis')
    plt.plot(sensor.getTimestamps(), sensor.getAccZCompensated())

    plt.subplot(3, 2, 2)
    plt.title(f'Filtered acceleration in X Axis (Cutoff freq.: {CUTOFF_FREQ}Hz)')
    plt.plot(sensor.getTimestamps(), butter_lowpass_filter(sensor.getAccXCompensated(), CUTOFF_FREQ, SAMPLING_FREQ, ORDER))
    plt.subplot(3, 2, 4)
    plt.title(f'Filtered acceleration in Y Axis (Cutoff freq.: {CUTOFF_FREQ}Hz)')
    plt.plot(sensor.getTimestamps(), butter_lowpass_filter(sensor.getAccYCompensated(), CUTOFF_FREQ, SAMPLING_FREQ, ORDER))
    plt.subplot(3, 2, 6)
    plt.title(f'Filtered acceleration in Z Axis (Cutoff freq.: {CUTOFF_FREQ}Hz)')
    plt.plot(sensor.getTimestamps(), butter_lowpass_filter(sensor.getAccZCompensated(), CUTOFF_FREQ, SAMPLING_FREQ, ORDER))

    plt.show()

def plotAnimatedAccelerationEstimated(sensor, filtered=False):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    XAxis, = ax.plot3D([-2, 2], [0, 0], [0, 0], 'k-', linewidth=0.5)
    YAxis, = ax.plot3D([0, 0], [-2, 2], [0, 0], 'k-', linewidth=0.5)
    ZAxis, = ax.plot3D([0, 0], [0, 0], [-2, 2], 'k-', linewidth=0.5)
    Rx, = ax.plot3D([], [], [], 'b-', linewidth=2)
    RxP1, = ax.plot3D([], [], [], 'k--', linewidth=1)
    RxP2, = ax.plot3D([], [], [], 'r--', linewidth=2)
    RxP3, = ax.plot3D([], [], [], 'g--', linewidth=4)
    Ry, = ax.plot3D([], [], [], 'r-', linewidth=2)
    RyP1, = ax.plot3D([], [], [], 'k--', linewidth=1)
    RyP2, = ax.plot3D([], [], [], 'k--', linewidth=1)
    RyP3, = ax.plot3D([], [], [], 'k--', linewidth=1)
    Rz, = ax.plot3D([], [], [], 'g-', linewidth=2)
    RzP1, = ax.plot3D([], [], [], 'k--', linewidth=1)
    RzP2, = ax.plot3D([], [], [], 'k--', linewidth=1)
    RzP3, = ax.plot3D([], [], [], 'k--', linewidth=1)
    R, = ax.plot3D([], [], [], 'k-', linewidth=3)

    ax.axes.set_xlim3d(left=-2, right=2)
    ax.axes.set_ylim3d(bottom=-2, top=2)
    ax.axes.set_zlim3d(bottom=-2, top=2)

    if not filtered:
        ani = animation.FuncAnimation(
            fig,
            update3D,
            len(sensor.getTimestamps()),
            fargs=(
                sensor.getRxEst(),
                sensor.getRyEst(),
                sensor.getRzEst(),
                [Rx, RxP1, RxP2, RxP3],
                [Ry, RyP1, RyP2, RyP3],
                [Rz, RzP1, RzP2, RzP3],
                R
            ),
            interval=10000 / len(sensor.getTimestamps()),
            blit=False
        )
    else:
        ani = animation.FuncAnimation(
            fig,
            update3D,
            len(sensor.getTimestamps()),
            fargs=(
                butter_lowpass_filter(sensor.getRxEst(), CUTOFF_FREQ, SAMPLING_FREQ, ORDER),
                butter_lowpass_filter(sensor.getRyEst(), CUTOFF_FREQ, SAMPLING_FREQ, ORDER),
                butter_lowpass_filter(sensor.getRzEst(), CUTOFF_FREQ, SAMPLING_FREQ, ORDER),
                [Rx, RxP1, RxP2, RxP3],
                [Ry, RyP1, RyP2, RyP3],
                [Rz, RzP1, RzP2, RzP3],
                R
            ),
            interval=10000 / len(sensor.getTimestamps()),
            blit=False
        )

    plt.show()

def plotAnimatedAccelerationCompensated(sensor, filtered=False):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    XAxis, = ax.plot3D([-2, 2], [0, 0], [0, 0], 'k-', linewidth=0.5)
    YAxis, = ax.plot3D([0, 0], [-2, 2], [0, 0], 'k-', linewidth=0.5)
    ZAxis, = ax.plot3D([0, 0], [0, 0], [-2, 2], 'k-', linewidth=0.5)
    Rx, = ax.plot3D([], [], [], 'b-', linewidth=2)
    RxP1, = ax.plot3D([], [], [], 'k--', linewidth=1)
    RxP2, = ax.plot3D([], [], [], 'r--', linewidth=2)
    RxP3, = ax.plot3D([], [], [], 'g--', linewidth=4)
    Ry, = ax.plot3D([], [], [], 'r-', linewidth=2)
    RyP1, = ax.plot3D([], [], [], 'k--', linewidth=1)
    RyP2, = ax.plot3D([], [], [], 'k--', linewidth=1)
    RyP3, = ax.plot3D([], [], [], 'k--', linewidth=1)
    Rz, = ax.plot3D([], [], [], 'g-', linewidth=2)
    RzP1, = ax.plot3D([], [], [], 'k--', linewidth=1)
    RzP2, = ax.plot3D([], [], [], 'k--', linewidth=1)
    RzP3, = ax.plot3D([], [], [], 'k--', linewidth=1)
    R, = ax.plot3D([], [], [], 'k-', linewidth=3)

    ax.axes.set_xlim3d(left=-2, right=2)
    ax.axes.set_ylim3d(bottom=-2, top=2)
    ax.axes.set_zlim3d(bottom=-2, top=2)

    if not filtered:
        ani = animation.FuncAnimation(
            fig,
            update3D,
            len(sensor.getTimestamps()),
            fargs=(
                sensor.getAccXCompensated(),
                sensor.getAccYCompensated(),
                sensor.getAccZCompensated(),
                [Rx, RxP1, RxP2, RxP3],
                [Ry, RyP1, RyP2, RyP3],
                [Rz, RzP1, RzP2, RzP3],
                R
            ),
            interval=10000 / len(sensor.getTimestamps()),
            blit=False
        )
    else:
        ani = animation.FuncAnimation(
            fig,
            update3D,
            len(sensor.getTimestamps()),
            fargs=(
                butter_lowpass_filter(sensor.getAccXCompensated(), CUTOFF_FREQ, SAMPLING_FREQ, ORDER),
                butter_lowpass_filter(sensor.getAccYCompensated(), CUTOFF_FREQ, SAMPLING_FREQ, ORDER),
                butter_lowpass_filter(sensor.getAccZCompensated(), CUTOFF_FREQ, SAMPLING_FREQ, ORDER),
                [Rx, RxP1, RxP2, RxP3],
                [Ry, RyP1, RyP2, RyP3],
                [Rz, RzP1, RzP2, RzP3],
                R
            ),
            interval=10000 / len(sensor.getTimestamps()),
            blit=False
        )

    plt.show()

def plotAccelerationAndVelocity(sensor):
    plt.subplot(3, 2, 1)
    plt.title('Compensated acceleration in X Axis')
    plt.plot(sensor.getTimestamps(), sensor.getAccXCompensated())
    plt.subplot(3, 2, 3)
    plt.title('Compensated acceleration in Y Axis')
    plt.plot(sensor.getTimestamps(), sensor.getAccYCompensated())
    plt.subplot(3, 2, 5)
    plt.title('Compensated acceleration in Z Axis')
    plt.plot(sensor.getTimestamps(), sensor.getAccZCompensated())
    plt.subplot(3, 2, 2)
    plt.title('Velocity in X Axis')
    plt.plot(sensor.getTimestamps(), sensor.getVelocityX())
    plt.subplot(3, 2, 4)
    plt.title('Velocity in Y Axis')
    plt.plot(sensor.getTimestamps(), sensor.getVelocityY())
    plt.subplot(3, 2, 6)
    plt.title('Velocity in Z Axis')
    plt.plot(sensor.getTimestamps(), sensor.getVelocityZ())

    plt.show()

def plotAccelerationAndVelocityFiltered(sensor):
    plt.subplot(3, 2, 1)
    plt.title('Compensated acceleration in X Axis')
    plt.plot(sensor.getTimestamps(), butter_lowpass_filter(sensor.getAccXCompensated(), CUTOFF_FREQ, SAMPLING_FREQ, ORDER))
    plt.subplot(3, 2, 3)
    plt.title('Compensated acceleration in Y Axis')
    plt.plot(sensor.getTimestamps(), butter_lowpass_filter(sensor.getAccYCompensated(), CUTOFF_FREQ, SAMPLING_FREQ, ORDER))
    plt.subplot(3, 2, 5)
    plt.title('Compensated acceleration in Z Axis')
    plt.plot(sensor.getTimestamps(), butter_lowpass_filter(sensor.getAccZCompensated(), CUTOFF_FREQ, SAMPLING_FREQ, ORDER))
    plt.subplot(3, 2, 2)
    plt.title('Velocity in X Axis')
    plt.plot(sensor.getTimestamps(), butter_lowpass_filter(sensor.getVelocityX(), CUTOFF_FREQ, SAMPLING_FREQ, ORDER))
    plt.subplot(3, 2, 4)
    plt.title('Velocity in Y Axis')
    plt.plot(sensor.getTimestamps(), butter_lowpass_filter(sensor.getVelocityY(), CUTOFF_FREQ, SAMPLING_FREQ, ORDER))
    plt.subplot(3, 2, 6)
    plt.title('Velocity in Z Axis')
    plt.plot(sensor.getTimestamps(), butter_lowpass_filter(sensor.getVelocityZ(), CUTOFF_FREQ, SAMPLING_FREQ, ORDER))

    plt.show()

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

    # plotRawMeasurements(sensor)
    # plotCalculation(sensor)
    # plotComparison(sensor)
    # plotFiltering(sensor)
    # plotAnimatedAccelerationEstimated(sensor, filtered=True)
    # plotAnimatedAccelerationCompensated(sensor, filtered=True)
    # plotAccelerationAndVelocity(sensor)
    plotAccelerationAndVelocityFiltered(sensor)