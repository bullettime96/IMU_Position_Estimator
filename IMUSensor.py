from DataPacket import DataPacket as dp
import numpy as np


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

        print(f"Sampling period: {SamplingPeriod}")
        RxAcc = self.getRawAcceleratorX()
        gyroX = self.getGyroX()
        RyAcc = self.getRawAcceleratorY()
        gyroY = self.getGyroY()
        RzAcc = self.getRawAcceleratorZ()
        gyroZ = self.getGyroZ()
        RAcc = [np.sqrt((RxAcc[0] ** 2) + (RyAcc[0] ** 2) + (RzAcc[0] ** 2))]
        RxAccComp = [RxAcc[0]]
        RyAccComp = [RyAcc[0]]
        RzAccComp = [RzAcc[0] - 1]

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
            RzAccComp[-1] -= 1 #Gravity compensation

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

        print(f"VelX len: {len(VelX)}")
        print(f"Times len: {len(times)}")
        for i in range(1, len(self.getAccXCompensated())):
            VelX.append(VelX[i - 1] + (self.AccComp[0][i] * SamplingPeriod))
            VelY.append(VelY[i - 1] + (self.AccComp[1][i] * SamplingPeriod))
            VelZ.append(VelZ[i - 1] + (self.AccComp[2][i] * SamplingPeriod))
            print(f"VelX len: {len(VelX)}")

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
