import struct
import time
import serial
import os
import signal
import sys
from datetime import datetime
import logging

dataDevice = None 
cliDevice = None 

# Constants
CLI_SERIAL_PORT = '/dev/ttyUSB0'
DATA_SERIAL_PORT = '/dev/ttyUSB1'
RADAR_CFG_FILE = '/home/ttpl2rhs/Desktop/Radar_test/profiles_50_20.cfg'
HEADER_LENGTH = 40
LOG_DIR = f'/home/ttpl2rhs/Desktop/Radar_test/logs/{time.strftime("%Y-%m-%d")}'
CSV_DIR = f'/home/ttpl2rhs/Desktop/Radar_test/CSV_Files/{time.strftime("%Y-%m-%d")}'

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

def configure_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s]- %(message)s',
        filename=os.path.join(LOG_DIR, log_file),
    )

def signal_handler(signal, frame):
    global dataDevice, cliDevice
    msg = "Received KeyboardInterrupt, closing dataDevice..."
    logging.info(msg)
    if dataDevice is not None:
        try:
            dataDevice.flushInput()
            dataDevice.flushOutput()
        except:
            pass
        time.sleep(2)
        dataDevice.close()
    if cliDevice is not None:
        try:
            cliDevice.flushInput()
            cliDevice.flushOutput()
        except:
            pass
        time.sleep(2)
        cliDevice.close()
    sys.exit(0)

def tlv_header_decode(data):
    if len(data) >= 8:
        tlv_type, tlv_length = struct.unpack('2I', data[:8])
        return tlv_type, tlv_length
    else:
        msg = "There are not enough bytes to unpack "
        logging.info(msg)
        return 0

def parse_detected_objects(data, tlv_length, num_detected_obj, frame_num, result):
    for i in range(num_detected_obj):
        x, y, z, vel = struct.unpack('4f', data[16 * i:16 * i + 16])
        current_time = time.time()
        milliseconds = int((current_time - int(current_time)) * 1000)
        timestamp = datetime.fromtimestamp(current_time).strftime("%H:%M:%S") + f":{milliseconds:03d}"
        result.write(f"{frame_num}, {i}, {x}, {y}, {z}, {vel}, {timestamp}\n")

def flush_serial_ports():
    global dataDevice, cliDevice
    for device in [dataDevice, cliDevice]:
        if device is not None:
            device.flushInput()
            device.flushOutput()

def create_initial_file():
    filename = f"data_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    file_path = os.path.join(CSV_DIR, filename)
    with open(file_path, 'w') as result:
        result.write("FrameNum,#DetOBj,X,Y,Z,Velocity,Time\n")
    return file_path

def main():
    global dataDevice, cliDevice
    file_path = None
    try:
        cliDevice = serial.Serial(CLI_SERIAL_PORT, 115200, timeout=3.0)
        dataDevice = serial.Serial(DATA_SERIAL_PORT, 921600, timeout=3.0)

        flush_serial_ports()  # Flush serial ports

        # Register the signal handler
        signal.signal(signal.SIGINT, signal_handler)

        # Write *.cfg file to the CLI/User port of the sensor
        cliDevice.write(('\r').encode())
        for line in open(RADAR_CFG_FILE):
            cliDevice.write(line.encode())
            temp = cliDevice.readline()
            if b"Ignored" in temp:
                msg = cliDevice.readline()
                logging.info(msg)
            time.sleep(0.05)
        cliDevice.close()

        file_path = create_initial_file()
        start_time = time.time()
        interval = 300

        while True:
            header = dataDevice.read(HEADER_LENGTH)
            magic, version, length, platform, frame_num, cpu_cycles, num_obj, num_tlvs, sub_frame_num = struct.unpack(
                'Q8I', header)
            # if magic is wrong, cycle through bytes until data stream is correct
            while magic != 506660481457717506:
                header = header[1:] + dataDevice.read(1)
                magic, version, length, platform, frame_num, cpu_cycles, num_obj, num_tlvs, sub_frame_num = struct.unpack(
                    'Q8I', header)
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time >= interval:
                start_time = time.time()
                filename = f"data_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
                file_path = os.path.join(CSV_DIR, filename)
                current_date = time.strftime('%Y-%m-%d')
                # Specify the directory path
                directory_path = f"/home/ttpl2rhs/Desktop/Radar_test/CSV_Files/{current_date}"
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                file_path = os.path.join(directory_path, filename)
                result.close()
                result = open(file_path, "w", 1)
                result.write("FrameNum,#DetOBj,X,Y,Z,Velocity,Time\n")

            if file_path:
                with open(file_path, 'a') as result:
                    initial_data = dataDevice.read(length - HEADER_LENGTH)
                    data = header + initial_data
                    if not data:
                        break
                    header = data[:HEADER_LENGTH]
                    try:
                        magic, version, total_length, platform, frame_num, cpu_cycles, num_obj, num_tlvs, sub_frame_num = struct.unpack(
                            'Q8I', header)
                    except:
                        msg = "Improper TLV structure found: " + str((data,))
                        logging.error(msg)
                        break
                    pending_bytes = total_length - HEADER_LENGTH
                    data = data[HEADER_LENGTH:]
                    for i in range(num_tlvs):
                        tlv_type, tlv_length = tlv_header_decode(data[:8])
                        data = data[8:]
                        if tlv_type == 1:
                            parse_detected_objects(data, tlv_length, num_obj, frame_num, result)
                        elif tlv_type in [2, 6, 7, 9]:
                            pass  
                        else:
                            msg = f"Unidentified tlv type {tlv_type}"
                            logging.error(msg)
                        data = data[tlv_length:]
                        pending_bytes -= (8 + tlv_length)
                    data = data[pending_bytes:]
    except serial.SerialException as e:
        msg = 'An Exception Occurred'
        logging.error(msg)

        msg = f'Exception Details: {e}'
        logging.error(msg)
    finally:
        msg = "Finally Abrupt Termination, closing dataDevice..."
        logging.info(msg)
        
        if dataDevice is not None:
            try:
                dataDevice.flushInput()
                dataDevice.flushOutput()
            except:
                pass
            dataDevice.close()
        if cliDevice is not None:
            try:
                cliDevice.flushInput()
                cliDevice.flushOutput()                
            except:
                pass
            cliDevice.close()
        if file_path:
            logging.info(f"Closing file: {file_path}")
            # Explicitly close the file to ensure data is saved before exiting
            try:
                with open(file_path, 'a') as result:
                    result.close()
            except Exception as e:
                logging.error(f"Error closing file: {e}")

if __name__ == "__main__":
    configure_logging(f'script_log_rhs{datetime.now().strftime("%Y%m%d%H%M%S")}.log')
    main()
