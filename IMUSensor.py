from DataPacket import DataPacket as dp
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from matplotlib import animation


class IMUSensor:
    def __init__(self, name="IMU", rawData=None, rawDataFormat=None):
        self.name = name
        self.rawData = []
        if not (rawData is None):
            for row in rawData:
                self.rawData.append(row)
        if not (rawDataFormat is None):
            self.rawDataFormat = rawDataFormat
        else:
            self.rawDataFormat = []
        self.dataPackets = []
        self.RAcc = []
        self.RGyro = []
        self.REst = []
        self.R = []
        self.AccComp = []
        self.Velocity = []
        self.Trajectory = []

    def parse(self, rawData=None, rawDataFormat=None):
        if not (rawData is None):
            self.rawData = []
            for row in rawData:
                self.rawData.append(row)
        if not (rawDataFormat is None):
            self.rawDataFormat = rawDataFormat

        self.dataPackets = []
        packetIdx = 0
        for packet in self.rawData:
            try:
                self.dataPackets.append(dp(packet, self.rawDataFormat))
            except dp.DataLengthException as exc:
                print(f"Data length exception on row {packetIdx + 1}")
                print(exc)
            except dp.DataTypeException as exc:
                print("Data type error in specified format.")
                print(exc)
                return []
            except dp.InvalidElementException as exc:
                print("Data element error in specified format.")
                print(exc)
                return []

        self.__normalize_timestamps__()
        return self.dataPackets

    def calculateAcceleration(self):

        times = self.getTimestamps() / 1000
        times = np.array([times[i] - times[i - 1] for i in range(1, len(times))])
        SamplingPeriod = np.mean(times)

        RxAcc = self.getRawAcceleratorX()
        gyroX = self.getGyroX()
        RyAcc = self.getRawAcceleratorY()
        gyroY = self.getGyroY()
        RzAcc = self.getRawAcceleratorZ()
        gyroZ = self.getGyroZ()
        RAcc = [np.sqrt((RxAcc[0] ** 2) + (RyAcc[0] ** 2) + (RzAcc[0] ** 2))]
        RxAccComp = [RxAcc[0]]
        RyAccComp = [RyAcc[0]]
        RzAccComp = [RzAcc[0]]
        # RzAccComp = [RzAcc[0] - 1]

        RxAcc[0] /= RAcc[0]
        RyAcc[0] /= RAcc[0]
        RzAcc[0] /= RAcc[0]

        Axy = [np.arctan2(RxAcc[0], RyAcc[0])]
        Axz = [np.arctan2(RxAcc[0], RzAcc[0])]
        Ayz = [np.arctan2(RyAcc[0], RzAcc[0])]
        RxGyro = [0]
        RyGyro = [0]
        RzGyro = [1]
        RxEst = [RxAcc[0]]
        RyEst = [RyAcc[0]]
        RzEst = [RzAcc[0]]

        wGyro = 8

        for i in range(1, len(self.dataPackets)):
            RAcc.append(np.sqrt((RxAcc[i] ** 2) + (RyAcc[i] ** 2) + (RzAcc[i] ** 2)))
            RxAcc[i] /= RAcc[i]
            RyAcc[i] /= RAcc[i]
            RzAcc[i] /= RAcc[i]

            Axy.append(np.arctan2(RxEst[i - 1], RyEst[i - 1]) + (((gyroZ[i - 1] + gyroZ[i]) / 2) * SamplingPeriod))
            Axz.append(np.arctan2(RxEst[i - 1], RzEst[i - 1]) + (((gyroY[i - 1] + gyroY[i]) / 2) * SamplingPeriod))
            Ayz.append(np.arctan2(RyEst[i - 1], RzEst[i - 1]) + (((gyroX[i - 1] + gyroX[i]) / 2) * SamplingPeriod))

            # Axy.append(np.arctan2(RxEst[i - 1], RyEst[i - 1]) + (gyroZ[i - 1] * SamplingPeriod))
            # Axz.append(np.arctan2(RxEst[i - 1], RzEst[i - 1]) + (gyroY[i - 1] * SamplingPeriod))
            # Ayz.append(np.arctan2(RyEst[i - 1], RzEst[i - 1]) + (gyroX[i - 1] * SamplingPeriod))

            # Axy.append(Axy[i - 1] + (((gyroZ[i - 1] + gyroZ[i]) / 2) * SamplingPeriod))
            # Axz.append(Axz[i - 1] + (((gyroY[i - 1] + gyroY[i]) / 2) * SamplingPeriod))
            # Ayz.append(Ayz[i - 1] + (((gyroX[i - 1] + gyroX[i]) / 2) * SamplingPeriod))

            RxGyro.append(np.sin(Axz[i]) / np.sqrt(1 + ((np.cos(Axz[i]) ** 2) * (np.tan(Ayz[i]) ** 2))))
            RyGyro.append(np.sin(Ayz[i]) / np.sqrt(1 + ((np.cos(Ayz[i]) ** 2) * (np.tan(Axz[i]) ** 2))))
            RzGyro.append(np.cos(Ayz[i]) / np.sqrt(1 + ((np.cos(Ayz[i]) ** 2) * (np.tan(Axz[i]) ** 2))))

            RxEst.append((RxAcc[i] + (RxGyro[i] * wGyro)) / (1 + wGyro))
            RyEst.append((RyAcc[i] + (RyGyro[i] * wGyro)) / (1 + wGyro))
            RzEst.append((RzAcc[i] + (RzGyro[i] * wGyro)) / (1 + wGyro))

            REst = np.sqrt((RxEst[i] ** 2) + (RyEst[i] ** 2) + (RzEst[i] ** 2))

            RxEst[i] /= REst
            RyEst[i] /= REst
            RzEst[i] /= REst

            RxAccComp.append(RxEst[i] * RAcc[i])
            RyAccComp.append(RyEst[i] * RAcc[i])
            RzAccComp.append(RzEst[i] * RAcc[i])
            # RzAccComp[-1] -= 1 #Gravity compensation

        self.RAcc = [RxAcc, RyAcc, RzAcc]
        self.RGyro = [RxGyro, RyGyro, RzGyro]
        self.REst = [RxEst, RyEst, RzEst]
        self.R = RAcc
        self.AccComp = [RxAccComp, RyAccComp, RzAccComp]

        return self.REst

    def calculateVelocity(self, initialVelocity=None):
        if initialVelocity is None:
            initialVelocity = [0, 0, 0]
        if not self.AccComp:
            return self.Velocity

        VelX = [initialVelocity[0]]
        VelY = [initialVelocity[1]]
        VelZ = [initialVelocity[2]]

        times = self.getTimestamps() / 1000
        times = np.array([times[i] - times[i - 1] for i in range(1, len(times))])
        SamplingPeriod = np.mean(times)

        for i in range(1, len(self.getAccXCompensated())):
            VelX.append(VelX[i - 1] + (self.AccComp[0][i] * SamplingPeriod))
            VelY.append(VelY[i - 1] + (self.AccComp[1][i] * SamplingPeriod))
            VelZ.append(VelZ[i - 1] + (self.AccComp[2][i] * SamplingPeriod))

        self.Velocity = [VelX, VelY, VelZ]
        return self.Velocity

    def calculateTrajectory(self, startingPoint=None):
        if startingPoint is None:
            startingPoint = [0, 0, 0]
        if not self.Velocity:
            return self.Trajectory

        Sx = [startingPoint[0]]
        Sy = [startingPoint[1]]
        Sz = [startingPoint[2]]

        times = self.getTimestamps() / 1000
        times = np.array([times[i] - times[i - 1] for i in range(1, len(times))])
        SamplingPeriod = np.mean(times)

        for i in range(1, len(times)):
            Sx.append(Sx[i - 1] + (((self.Velocity[0][i - 1] + self.Velocity[0][i]) / 2) * SamplingPeriod))
            Sy.append(Sy[i - 1] + (((self.Velocity[1][i - 1] + self.Velocity[1][i]) / 2) * SamplingPeriod))
            Sz.append(Sz[i - 1] + (((self.Velocity[2][i - 1] + self.Velocity[2][i]) / 2) * SamplingPeriod))

        self.Trajectory = [Sx, Sy, Sz]
        return self.Trajectory

    def __normalize_timestamps__(self):
        minStamp = np.min([self.dataPackets[i].timestamp for i in range(0, len(self.dataPackets))])

        for i in range(0, len(self.dataPackets)):
            self.dataPackets[i].timestamp -= minStamp

    def getTimestamps(self):
        return np.array([self.dataPackets[i].timestamp for i in range(0, len(self.dataPackets))])

    def getRawAcceleratorComplete(self):
        return np.array([
            [self.dataPackets[i].accX for i in range(0, len(self.dataPackets))],
            [self.dataPackets[i].accY for i in range(0, len(self.dataPackets))],
            [self.dataPackets[i].accZ for i in range(0, len(self.dataPackets))]
        ]
        )

    def getRawAcceleratorX(self):
        return np.array([self.dataPackets[i].accX for i in range(0, len(self.dataPackets))])

    def getRawAcceleratorY(self):
        return np.array([self.dataPackets[i].accY for i in range(0, len(self.dataPackets))])

    def getRawAcceleratorZ(self):
        return np.array([self.dataPackets[i].accZ for i in range(0, len(self.dataPackets))])

    def getGyroX(self):
        return np.array([self.dataPackets[i].gyroX for i in range(0, len(self.dataPackets))])

    def getGyroY(self):
        return np.array([self.dataPackets[i].gyroY for i in range(0, len(self.dataPackets))])

    def getGyroZ(self):
        return np.array([self.dataPackets[i].gyroZ for i in range(0, len(self.dataPackets))])

    def getGyroComplete(self):
        return np.array([
            [self.dataPackets[i].gyroX for i in range(0, len(self.dataPackets))],
            [self.dataPackets[i].gyroY for i in range(0, len(self.dataPackets))],
            [self.dataPackets[i].gyroZ for i in range(0, len(self.dataPackets))]
        ]
        )

    def getRxGyro(self):
        if self.RGyro:
            return self.RGyro[0]
        else:
            return []

    def getRyGyro(self):
        if self.RGyro:
            return self.RGyro[1]
        else:
            return []

    def getRzGyro(self):
        if self.RGyro:
            return self.RGyro[2]
        else:
            return []

    def getRGyro(self):
        return self.RGyro

    def getRxAcc(self):
        if self.RAcc:
            return self.RAcc[0]
        else:
            return []

    def getRyAcc(self):
        if self.RAcc:
            return self.RAcc[1]
        else:
            return []

    def getRzAcc(self):
        if self.RAcc:
            return self.RAcc[2]
        else:
            return []

    def getRAcc(self):
        return self.RAcc

    def getRxEst(self):
        if self.REst:
            return self.REst[0]
        else:
            return []

    def getRyEst(self):
        if self.REst:
            return self.REst[1]
        else:
            return []

    def getRzEst(self):
        if self.REst:
            return self.REst[2]
        else:
            return []

    def getREst(self):
        return self.REst

    def getAccXCompensated(self):
        return self.AccComp[0]

    def getAccYCompensated(self):
        return self.AccComp[1]

    def getAccZCompensated(self):
        return self.AccComp[2]

    def getAccCompensated(self):
        return self.AccComp

    def getTotalAcceleration(self):
        return self.R

    def getVelocityX(self):
        if self.Velocity:
            return self.Velocity[0]
        else:
            return []

    def getVelocityY(self):
        if self.Velocity:
            return self.Velocity[1]
        else:
            return []

    def getVelocityZ(self):
        if self.Velocity:
            return self.Velocity[2]
        else:
            return []

    def getVelocity(self):
        return self.Velocity

    def getTrajectoryX(self):
        if self.Trajectory:
            return self.Trajectory[0]
        else:
            return []

    def getTrajectoryY(self):
        if self.Trajectory:
            return self.Trajectory[1]
        else:
            return []

    def getTrajectoryZ(self):
        if self.Trajectory:
            return self.Trajectory[2]
        else:
            return []

    def getTrajectory(self):
        return self.Trajectory

    def __repr__(self) -> str:
        if not self.dataPackets:
            return f"IMU Sensor (name: {self.name}, Raw data length: {len(self.rawData)}, Data format: " \
                   f"{self.rawDataFormat})"
        elif not self.AccComp:
            return f"IMU Sensor (name: {self.name}, Raw data length: {len(self.rawData)}, Data format: " \
                   f"{self.rawDataFormat}, Parsed data length: {len(self.dataPackets)})"
        elif not self.Velocity:
            return f"IMU Sensor (name: {self.name}, Raw data length: {len(self.rawData)}, Data format: " \
                f"{self.rawDataFormat}, Parsed data length: {len(self.dataPackets)}, " \
                f"Acceleration data length: {len(self.getAccXCompensated())})"
        elif not self.Trajectory:
            return f"IMU Sensor (name: {self.name}, Raw data length: {len(self.rawData)}, Data format: " \
                f"{self.rawDataFormat}, Parsed data length: {len(self.dataPackets)}, " \
                f"Acceleration data length: {len(self.getAccXCompensated())}, Velocity data length: " \
                   f"{len(self.getVelocityX())})"
        else:
            return f"IMU Sensor (name: {self.name}, Raw data length: {len(self.rawData)}, Data format: " \
                f"{self.rawDataFormat}, Parsed data length: {len(self.dataPackets)}, " \
                f"Acceleration data length: {len(self.getAccXCompensated())}, Velocity data length: " \
                   f"{len(self.getVelocityX())}, Trajectory data length: {len(self.getTrajectoryX())})"

    def __butter_lowpass(self, cutoff, fs, order=5):
        return butter(order, cutoff, fs=fs, btype='low', analog=False)

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.__butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def update3D(self, iteration, RxData, RyData, RzData, RxLines, RyLines, RzLines, RLine):
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
        RzLines[3].set_data(np.array([RxData[iteration], RxData[iteration]]),
                            np.array([RyData[iteration], RyData[iteration]]))
        RzLines[3].set_3d_properties(np.array([0, RzData[iteration]]))

        RLine.set_data(np.array([0, RxData[iteration]]), np.array([0, RyData[iteration]]))
        RLine.set_3d_properties(np.array([0, RzData[iteration]]))

    def plotRawMeasurements(self, show=False, file='rawMeasurements.jpg'):
        plt.figure(figsize=(20, 25), dpi=300)
        plt.subplot(4, 2, 1)
        plt.title('Acceleration in X Axis')
        plt.plot(self.getTimestamps(), self.getRawAcceleratorX())
        plt.subplot(4, 2, 3)
        plt.title('Acceleration in Y Axis')
        plt.plot(self.getTimestamps(), self.getRawAcceleratorY())
        plt.subplot(4, 2, 5)
        plt.title('Acceleration in Z Axis')
        plt.plot(self.getTimestamps(), self.getRawAcceleratorZ())
        plt.subplot(4, 2, 7)
        plt.title('Total Acceleration')
        plt.plot(self.getTimestamps(), self.getTotalAcceleration())
        plt.subplot(4, 2, 2)
        plt.title('Rotation rate around X Axis')
        plt.plot(self.getTimestamps(), self.getGyroX())
        plt.subplot(4, 2, 4)
        plt.title('Rotation rate around Y Axis')
        plt.plot(self.getTimestamps(), self.getGyroY())
        plt.subplot(4, 2, 6)
        plt.title('Rotation rate around Z Axis')
        plt.plot(self.getTimestamps(), self.getGyroZ())
        if show:
            plt.show()
        else:
            plt.savefig(file)

    def plotCalculation(self, show=False, file='Calculations.jpg'):
        plt.figure(figsize=(20, 25), dpi=300)
        plt.subplot(3, 3, 1)
        plt.title('Normalized raw acceleration in X Axis')
        plt.plot(self.getTimestamps(), self.getRxAcc())
        plt.subplot(3, 3, 4)
        plt.title('Normalized raw acceleration in Y Axis')
        plt.plot(self.getTimestamps(), self.getRyAcc())
        plt.subplot(3, 3, 7)
        plt.title('Normalized raw acceleration in Z Axis')
        plt.plot(self.getTimestamps(), self.getRzAcc())
        plt.subplot(3, 3, 2)

        plt.title('Compensating acceleration from Gyro in X Axis')
        plt.plot(self.getTimestamps(), self.getRxGyro())
        plt.subplot(3, 3, 5)
        plt.title('Compensating acceleration from Gyro in Y Axis')
        plt.plot(self.getTimestamps(), self.getRyGyro())
        plt.subplot(3, 3, 8)
        plt.title('Compensating acceleration from Gyro in Z Axis')
        plt.plot(self.getTimestamps(), self.getRzGyro())

        plt.subplot(3, 3, 3)
        plt.title('Normalized estimated acceleration in X Axis')
        plt.plot(self.getTimestamps(), self.getRxEst())
        plt.subplot(3, 3, 6)
        plt.title('Normalized estimated acceleration in Y Axis')
        plt.plot(self.getTimestamps(), self.getRyEst())
        plt.subplot(3, 3, 9)
        plt.title('Normalized estimated acceleration in Z Axis')
        plt.plot(self.getTimestamps(), self.getRzEst())

        if show:
            plt.show()
        else:
            plt.savefig(file)

    def plotComparison(self, show=False, file='Comparison.jpg'):
        plt.figure(figsize=(20, 25), dpi=300)
        plt.subplot(4, 3, 1)
        plt.title('Acceleration in X Axis')
        plt.plot(self.getTimestamps(), self.getRawAcceleratorX())
        plt.subplot(4, 3, 4)
        plt.title('Acceleration in Y Axis')
        plt.plot(self.getTimestamps(), self.getRawAcceleratorY())
        plt.subplot(4, 3, 7)
        plt.title('Acceleration in Y Axis')
        plt.plot(self.getTimestamps(), self.getRawAcceleratorZ())
        plt.subplot(4, 3, 10)
        plt.title('Total Acceleration')
        plt.plot(self.getTimestamps(), self.getTotalAcceleration())

        plt.subplot(4, 3, 2)
        plt.title('Normalized estimated acceleration in X Axis')
        plt.plot(self.getTimestamps(), self.getRxEst())
        plt.subplot(4, 3, 5)
        plt.title('Normalized estimated acceleration in Y Axis')
        plt.plot(self.getTimestamps(), self.getRyEst())
        plt.subplot(4, 3, 8)
        plt.title('Normalized estimated acceleration in Z Axis')
        plt.plot(self.getTimestamps(), self.getRzEst())

        plt.subplot(4, 3, 3)
        plt.title('Compensated acceleration in X Axis')
        plt.plot(self.getTimestamps(), self.getAccXCompensated())
        plt.subplot(4, 3, 6)
        plt.title('Compensated acceleration in Y Axis')
        plt.plot(self.getTimestamps(), self.getAccYCompensated())
        plt.subplot(4, 3, 9)
        plt.title('Compensated acceleration in Z Axis')
        plt.plot(self.getTimestamps(), self.getAccZCompensated())

        if show:
            plt.show()
        else:
            plt.savefig(file)

    def plotFiltering(self, show=False, file='Filtering.jpg', Cutoff_Freq=25, Sampling_Freq=333, Order=5):
        plt.figure(figsize=(20, 25), dpi=300)
        plt.subplot(3, 2, 1)
        plt.title('Compensated acceleration in X Axis')
        plt.plot(self.getTimestamps(), self.getAccXCompensated())
        plt.subplot(3, 2, 3)
        plt.title('Compensated acceleration in Y Axis')
        plt.plot(self.getTimestamps(), self.getAccYCompensated())
        plt.subplot(3, 2, 5)
        plt.title('Compensated acceleration in Z Axis')
        plt.plot(self.getTimestamps(), self.getAccZCompensated())

        plt.subplot(3, 2, 2)
        plt.title(f'Filtered acceleration in X Axis (Cutoff freq.: {Cutoff_Freq}Hz)')
        plt.plot(self.getTimestamps(),
                 self.butter_lowpass_filter(self.getAccXCompensated(), Cutoff_Freq, Sampling_Freq, Order))
        plt.subplot(3, 2, 4)
        plt.title(f'Filtered acceleration in Y Axis (Cutoff freq.: {Cutoff_Freq}Hz)')
        plt.plot(self.getTimestamps(),
                 self.butter_lowpass_filter(self.getAccYCompensated(), Cutoff_Freq, Sampling_Freq, Order))
        plt.subplot(3, 2, 6)
        plt.title(f'Filtered acceleration in Z Axis (Cutoff freq.: {Cutoff_Freq}Hz)')
        plt.plot(self.getTimestamps(),
                 self.butter_lowpass_filter(self.getAccZCompensated(), Cutoff_Freq, Sampling_Freq, Order))

        if show:
            plt.show()
        else:
            plt.savefig(file)

    def plotAnimatedAccelerationEstimated(self, filtered=False, Cutoff_Freq=25, Sampling_Freq=333, Order=5):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        XAxis, = ax.plot3D([-2, 2], [0, 0], [0, 0], 'k-', linewidth=0.5)
        YAxis, = ax.plot3D([0, 0], [-2, 2], [0, 0], 'k-', linewidth=0.5)
        ZAxis, = ax.plot3D([0, 0], [0, 0], [-2, 2], 'k-', linewidth=0.5)
        Rx, = ax.plot3D([], [], [], 'b-', linewidth=2)
        RxP1, = ax.plot3D([], [], [], 'k--', linewidth=1)
        RxP2, = ax.plot3D([], [], [], 'k--', linewidth=1)
        RxP3, = ax.plot3D([], [], [], 'k--', linewidth=1)
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
                self.update3D,
                len(self.getTimestamps()),
                fargs=(
                    self.getRxEst(),
                    self.getRyEst(),
                    self.getRzEst(),
                    [Rx, RxP1, RxP2, RxP3],
                    [Ry, RyP1, RyP2, RyP3],
                    [Rz, RzP1, RzP2, RzP3],
                    R
                ),
                interval=10000 / len(self.getTimestamps()),
                blit=False
            )
        else:
            ani = animation.FuncAnimation(
                fig,
                self.update3D,
                len(self.getTimestamps()),
                fargs=(
                    self.butter_lowpass_filter(self.getRxEst(), Cutoff_Freq, Sampling_Freq, Order),
                    self.butter_lowpass_filter(self.getRyEst(), Cutoff_Freq, Sampling_Freq, Order),
                    self.butter_lowpass_filter(self.getRzEst(), Cutoff_Freq, Sampling_Freq, Order),
                    [Rx, RxP1, RxP2, RxP3],
                    [Ry, RyP1, RyP2, RyP3],
                    [Rz, RzP1, RzP2, RzP3],
                    R
                ),
                interval=10000 / len(self.getTimestamps()),
                blit=False
            )

        plt.show()

    def plotAnimatedAccelerationCompensated(self, filtered=False, Cutoff_Freq=25, Sampling_Freq=333, Order=5):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        XAxis, = ax.plot3D([-2, 2], [0, 0], [0, 0], 'k-', linewidth=0.5)
        YAxis, = ax.plot3D([0, 0], [-2, 2], [0, 0], 'k-', linewidth=0.5)
        ZAxis, = ax.plot3D([0, 0], [0, 0], [-2, 2], 'k-', linewidth=0.5)
        Rx, = ax.plot3D([], [], [], 'b-', linewidth=2)
        RxP1, = ax.plot3D([], [], [], 'k--', linewidth=1)
        RxP2, = ax.plot3D([], [], [], 'k--', linewidth=1)
        RxP3, = ax.plot3D([], [], [], 'k--', linewidth=1)
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
                self.update3D,
                len(self.getTimestamps()),
                fargs=(
                    self.getAccXCompensated(),
                    self.getAccYCompensated(),
                    self.getAccZCompensated(),
                    [Rx, RxP1, RxP2, RxP3],
                    [Ry, RyP1, RyP2, RyP3],
                    [Rz, RzP1, RzP2, RzP3],
                    R
                ),
                interval=10000 / len(self.getTimestamps()),
                blit=False
            )
        else:
            ani = animation.FuncAnimation(
                fig,
                self.update3D,
                len(self.getTimestamps()),
                fargs=(
                    self.butter_lowpass_filter(self.getAccXCompensated(), Cutoff_Freq, Sampling_Freq, Order),
                    self.butter_lowpass_filter(self.getAccYCompensated(), Cutoff_Freq, Sampling_Freq, Order),
                    self.butter_lowpass_filter(self.getAccZCompensated(), Cutoff_Freq, Sampling_Freq, Order),
                    [Rx, RxP1, RxP2, RxP3],
                    [Ry, RyP1, RyP2, RyP3],
                    [Rz, RzP1, RzP2, RzP3],
                    R
                ),
                interval=10000 / len(self.getTimestamps()),
                blit=False
            )

        plt.show()

    def plotAccelerationAndVelocity(self, show=False, file='AccelerationAndVelocity.jpg'):
        plt.figure(figsize=(20, 25), dpi=300)
        plt.subplot(3, 2, 1)
        plt.title('Compensated acceleration in X Axis')
        plt.plot(self.getTimestamps(), self.getAccXCompensated())
        plt.subplot(3, 2, 3)
        plt.title('Compensated acceleration in Y Axis')
        plt.plot(self.getTimestamps(), self.getAccYCompensated())
        plt.subplot(3, 2, 5)
        plt.title('Compensated acceleration in Z Axis')
        plt.plot(self.getTimestamps(), self.getAccZCompensated())
        plt.subplot(3, 2, 2)
        plt.title('Velocity in X Axis')
        plt.plot(self.getTimestamps(), self.getVelocityX())
        plt.subplot(3, 2, 4)
        plt.title('Velocity in Y Axis')
        plt.plot(self.getTimestamps(), self.getVelocityY())
        plt.subplot(3, 2, 6)
        plt.title('Velocity in Z Axis')
        plt.plot(self.getTimestamps(), self.getVelocityZ())

        if show:
            plt.show()
        else:
            plt.savefig(file)

    def plotAccelerationAndVelocityFiltered(self, show=False, file='AccelerationAndVelocityFiltered.jpg', Cutoff_Freq=25, Sampling_Freq=333, Order=5):
        plt.figure(figsize=(20, 25), dpi=300)
        plt.subplot(3, 2, 1)
        plt.title('Compensated acceleration in X Axis')
        plt.plot(self.getTimestamps(),
                 self.butter_lowpass_filter(self.getAccXCompensated(), Cutoff_Freq, Sampling_Freq, Order))
        plt.subplot(3, 2, 3)
        plt.title('Compensated acceleration in Y Axis')
        plt.plot(self.getTimestamps(),
                 self.butter_lowpass_filter(self.getAccYCompensated(), Cutoff_Freq, Sampling_Freq, Order))
        plt.subplot(3, 2, 5)
        plt.title('Compensated acceleration in Z Axis')
        plt.plot(self.getTimestamps(),
                 self.butter_lowpass_filter(self.getAccZCompensated(), Cutoff_Freq, Sampling_Freq, Order))
        plt.subplot(3, 2, 2)
        plt.title('Velocity in X Axis')
        plt.plot(self.getTimestamps(),
                 self.butter_lowpass_filter(self.getVelocityX(), Cutoff_Freq, Sampling_Freq, Order))
        plt.subplot(3, 2, 4)
        plt.title('Velocity in Y Axis')
        plt.plot(self.getTimestamps(),
                 self.butter_lowpass_filter(self.getVelocityY(), Cutoff_Freq, Sampling_Freq, Order))
        plt.subplot(3, 2, 6)
        plt.title('Velocity in Z Axis')
        plt.plot(self.getTimestamps(),
                 self.butter_lowpass_filter(self.getVelocityZ(), Cutoff_Freq, Sampling_Freq, Order))

        if show:
            plt.show()
        else:
            plt.savefig(file)
