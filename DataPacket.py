class DataPacket:
    class DataLengthException(Exception):
        def __init__(self, message=""):
            self.message = message
            super().__init__(self.message)

    class DataTypeException(Exception):
        def __init__(self, f=""):
            self.message = f"Invalid data format: {f}"
            super().__init__(self.message)

    class InvalidElementException(Exception):
        def __init__(self, f=""):
            self.message = f"Invalid data type: {f}"
            super().__init__(self.message)

    def __format_constructor__(self, rawFormat):
        dataFormat = []
        for element in rawFormat:
            if element.lower() == 'timestamp':
                dataFormat.append(IMUDataElement.TimeStamp())
            elif element.lower() == 'accx':
                dataFormat.append(IMUDataElement.AccX())
            elif element.lower() == 'accy':
                dataFormat.append(IMUDataElement.AccY())
            elif element.lower() == 'accz':
                dataFormat.append(IMUDataElement.AccZ())
            elif element.lower() == 'gyrox':
                dataFormat.append(IMUDataElement.GyroX())
            elif element.lower() == 'gyroy':
                dataFormat.append(IMUDataElement.GyroY())
            elif element.lower() == 'gyroz':
                dataFormat.append(IMUDataElement.GyroZ())
            else:
                raise self.InvalidElementException(element)
        return dataFormat

    def __init__(self, rawData, packetFormat):
        self.timestamp = None
        self.accX = None
        self.accY = None
        self.accZ = None
        self.gyroX = None
        self.gyroY = None
        self.gyroZ = None
        self.dataFormat = self.__format_constructor__(packetFormat)

        if len(rawData) > len(self.dataFormat):
            raise self.DataLengthException(message=f"Raw data is longer ({len(rawData)} elements) than the specified "
                                                   f"data format ({len(packetFormat)} elements).")
        if len(rawData) < len(self.dataFormat):
            raise self.DataLengthException(message=f"Raw data is shorter ({len(rawData)} elements) than the specified "
                                                   f"data format ({len(packetFormat)} elements).")

        merged_list = [{"rawData": rawData[i], "format": self.dataFormat[i]} for i in range(0, len(packetFormat))]

        for dataPoint in merged_list:
            if dataPoint["format"].DataType == 'uint32':
                parsedData = int(dataPoint["rawData"])
            elif dataPoint["format"].DataType == 'float':
                parsedData = float(dataPoint["rawData"])
            else:
                raise self.DataTypeException(dataPoint["format"].DataType)

            if dataPoint["format"].Convert:
                parsedData *= 500/32768

            if dataPoint["format"].ElementType == 'timestamp':
                self.timestamp = parsedData
            elif dataPoint["format"].ElementType == 'accx':
                self.accX = parsedData
            elif dataPoint["format"].ElementType == 'accy':
                self.accY = parsedData
            elif dataPoint["format"].ElementType == 'accz':
                self.accZ = parsedData
            elif dataPoint["format"].ElementType == 'gyrox':
                self.gyroX = parsedData
            elif dataPoint["format"].ElementType == 'gyroy':
                self.gyroY = parsedData
            elif dataPoint["format"].ElementType == 'gyroz':
                self.gyroZ = parsedData
            else:
                raise self.InvalidElementException(dataPoint["format"].ElementType)

    def __repr__(self) -> str:
        return f"Data point (Timestamp: {self.timestamp}, accX: {self.accX}, accY: {self.accY}, accZ: {self.accZ}, " \
               f"gyroX: {self.gyroX}, gyroY: {self.gyroY}, gyroZ: {self.gyroZ})"


class IMUDataElement:
    class TimeStamp:
        def __init__(self):
            self.ElementType = 'timestamp'
            self.DataType = 'uint32'
            self.Convert = False
    class AccX:
        def __init__(self):
            self.ElementType = 'accx'
            self.DataType = 'float'
            self.Convert = False

    class AccY:
        def __init__(self):
            self.ElementType = 'accy'
            self.DataType = 'float'
            self.Convert = False

    class AccZ:
        def __init__(self):
            self.ElementType = 'accz'
            self.DataType = 'float'
            self.Convert = False

    class GyroX:
        def __init__(self):
            self.ElementType = 'gyrox'
            self.DataType = 'float'
            self.Convert = True

    class GyroY:
        def __init__(self):
            self.ElementType = 'gyroy'
            self.DataType = 'float'
            self.Convert = True

    class GyroZ:
        def __init__(self):
            self.ElementType = 'gyroz'
            self.DataType = 'float'
            self.Convert = True