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
            match element.lower():
                case 'timestamp':
                    dataFormat.append(IMUDataElement.TimeStamp())
                case 'accx':
                    dataFormat.append(IMUDataElement.AccX())
                case 'accy':
                    dataFormat.append(IMUDataElement.AccY())
                case 'accz':
                    dataFormat.append(IMUDataElement.AccZ())
                case 'gyrox':
                    dataFormat.append(IMUDataElement.GyroX())
                case 'gyroy':
                    dataFormat.append(IMUDataElement.GyroY())
                case 'gyroz':
                    dataFormat.append(IMUDataElement.GyroZ())
                case _:
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
            match dataPoint["format"].DataType:
                case 'uint32':
                    parsedData = int(dataPoint["rawData"])
                case 'float':
                    parsedData = float(dataPoint["rawData"])
                case _:
                    raise self.DataTypeException(dataPoint["format"].DataType)
            if dataPoint["format"].Convert:
                parsedData *= 500/32768
            match dataPoint["format"].ElementType:
                case 'timestamp':
                    self.timestamp = parsedData
                case 'accx':
                    self.accX = parsedData
                case 'accy':
                    self.accY = parsedData
                case 'accz':
                    self.accZ = parsedData
                case 'gyrox':
                    self.gyroX = parsedData
                case 'gyroy':
                    self.gyroY = parsedData
                case 'gyroz':
                    self.gyroZ = parsedData
                case _:
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
            self.Convert = True

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