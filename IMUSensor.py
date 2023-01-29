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

        times = self.getTimestamps()
        times = np.array([times[i - 1] - times[i] for i in range(1, len(times))])
        SamplingPeriod = np.mean(times)

        RxAcc = self.getRawAcceleratorX()
        gyroX = self.getGyroX()
        RyAcc = self.getRawAcceleratorY()
        gyroY = self.getGyroY()
        RzAcc = self.getRawAcceleratorZ()
        gyroZ = self.getGyroZ()
        RAcc = np.sqrt((RxAcc ** 2) + (RyAcc ** 2) + (RzAcc ** 2))

        RxAcc /= RAcc
        RyAcc /= RAcc
        RzAcc /= RAcc

        Axy = [np.arctan2(RxAcc[0], RyAcc[0])]
        Axz = [np.arctan2(RxAcc[0], RzAcc[0])]
        Ayz = [np.arctan2(RyAcc[0], RzAcc[0])]
        RxGyro = [0]
        RyGyro = [0]
        RzGyro = [1]
        RxEst = RxAcc
        RyEst = RyAcc
        RzEst = RzAcc
        wGyro = 5

        for i in range(1, len(self.dataPackets)):

            Axy.append(Axy[i - 1] + (((gyroZ[i - 1] + gyroZ[i]) / 2) * SamplingPeriod))
            Axz.append(Axy[i - 1] + (((gyroY[i - 1] + gyroY[i]) / 2) * SamplingPeriod))
            Ayz.append(Axy[i - 1] + (((gyroX[i - 1] + gyroX[i]) / 2) * SamplingPeriod))

            RxGyro.append(1 / np.sqrt(1 + (((1/np.tan(Axz[i])) ** 2) * ((1/np.cos(Axz[i])) ** 2))))

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

    def __repr__(self) -> str:
        return f"IMU Sensor (name: {self.name}, Raw data length: {len(self.rawData)}, Data format: " \
               f"{self.rawDataFormat}, Parsed data length: {len(self.dataPackets)})"
