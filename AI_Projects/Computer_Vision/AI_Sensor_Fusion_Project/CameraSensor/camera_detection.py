import csv
import cv2
import datetime
from datetime import datetime, timedelta
from math import dist
import logging
import numpy as np
import os
import os.path
import pandas as pd
import time

from models.detector import TRTEngine
from models.utils import blob, letterbox
from ntracker import *


class_list = open("class.txt", "r")
data = class_list.read()
class_list = data.split("\n")

# Invoking Tracker Function
ntracker = Tracker()
current_date = time.strftime("%Y-%m-%d")
log_dir = f"logs/{current_date}"
os.makedirs(log_dir, exist_ok=True)


def configure_logging(log_file):
    # log_handler = TimedRotatingFileHandler(filename=os.path.join(log_dir,log_file), when="s", interval=20)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s]- %(message)s",
        filename=os.path.join(log_dir, log_file),
    )


def main():
    enggine = TRTEngine("./engine/tensorrt/best_8n_fp16.engine")
    H, W = enggine.inp_info[0].shape[-2:]

    cap = cv2.VideoCapture("rtsp://")
    w, h = int(cap.get(3)), int(cap.get(4))
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    # print("Resolution: {}x{}".format(w, h))
    # print("FPS: {}".format(input_fps))

    fps = 0.0

    # Setting Region of Interest (ROI)
    area = [(163, 321), (162, 352), (935, 352), (936, 322)]

    vh = {}
    counter = []
    car_counter = []
    mav_counter = []
    lcv_counter = []
    bus_counter = []
    auto_counter = []
    tractor_counter = []
    bike_counter = []

    # MAV Axle Count

    axle_2_counter = []
    axle_3_counter = []
    axle_4_counter = []
    axle_5_counter = []
    axle_6_counter = []
    axle_7_plus_counter = []
    melcv_counter = []
    minilcv_counter = []

    def append_data_to_csv(
        cameraip,
        edgeip,
        location,
        timestamp,
        lat,
        long,
        count,
        car_count,
        axle_2_count,
        mav_count,
        bike_count,
        bus_count,
        lcv_count,
        auto_count,
        tractor_count,
        axle_3_count,
        axle_4_count,
        axle_5_count,
        axle_6_count,
        axle_7_plus_count,
        lane,
        side,
        direction,
    ):
        # Create directory with current date if it doesn't exist
        current_date = time.strftime("%Y-%m-%d")
        directory_path = f"{side}/{direction}/{lane}_Lane/data/{current_date}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Create a new CSV file with column headers if it doesn't exist
        csv_file_path = f"{directory_path}/{side}_{current_date}.csv"

        # check if file exists, if not create it
        if not os.path.isfile(csv_file_path):
            msg = "Successfully CSV file is created !"
            logging.info(msg)
            # print(msg)
            with open(csv_file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Camera_IP",
                        "EdgeBox_IP",
                        "Location",
                        "Timestamp",
                        "Latitude",
                        "Longitude",
                        "Lane",
                        "Total_Count",
                        "Car_Count",
                        "Truck_Axle_2_Count",
                        "MAV_Count",
                        "Bike_Count",
                        "Bus_Count",
                        "LCV_Count",
                        "Auto_Count",
                        "Tractor_Count",
                        "Axle_3_Count",
                        "Axle_4_Count",
                        "Axle_5_count",
                        "Axle_6_Count",
                        "Axle_7_plus_Count",
                    ]
                )
        msg = "Data is appending to the csv file........"
        logging.info(msg)
        with open(csv_file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    cameraip,
                    edgeip,
                    location,
                    timestamp,
                    lat,
                    long,
                    lane,
                    count,
                    car_count,
                    axle_2_count,
                    mav_count,
                    bike_count,
                    bus_count,
                    lcv_count,
                    auto_count,
                    tractor_count,
                    axle_3_count,
                    axle_4_count,
                    axle_5_count,
                    axle_6_count,
                    axle_7_plus_count,
                ]
            )

    def mav_count_screenshot(folder_path, axle_counter, frame, bbox_id, class_type):
        # Create folder with the current date
        folder_name = datetime.now().strftime("%Y-%m-%d")
        screenshots_folder = os.path.join(folder_path, folder_name)
        if not os.path.exists(screenshots_folder):
            os.makedirs(screenshots_folder)
        # Save screenshot
        screenshot_name = (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            + "_bbox_id_"
            + str(bbox_id)
            + "_"
            + class_type
            + ".jpg"
        )
        screenshot_path = os.path.join(screenshots_folder, screenshot_name)
        # cropped_frame = frame[y3:y4, x3:x4]
        # cv2.imwrite(screenshot_path, cropped_frame)
        cv2.imwrite(screenshot_path, frame)

        # Increase the count
        if axle_counter is not None:
            axle_counter.append(id)
        return axle_counter

    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            msg = "Error reading im from camera Waiting and retrying..."
            print(msg)
            logging.critical(msg)
            # print("Error reading im from camera. Waiting and retrying...")
            time.sleep(5)  # wait for 5 seconds
            cap = cv2.VideoCapture("rtsp://")
            continue  # go back to the beginning of the loop

        t1 = time.time()
        draw = frame.copy()
        bgr, ratio, dwdh = letterbox(frame, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)

        result = enggine(tensor)
        bboxes, scores, labels = result
        bboxes -= dwdh
        bboxes /= ratio
        bboxes_np = bboxes
        scores_np = scores
        labels_np = labels

        # Combine the arrays into a single NumPy array
        detections = np.concatenate(
            (bboxes_np, labels_np[:, np.newaxis], scores_np[:, np.newaxis]), axis=1
        )
        px = pd.DataFrame(detections).astype("float")

        # Bounding Box List:

        total_list = []
        car_list = []
        bus_list = []
        bike_list = []
        lcv_list = []
        mav_list = []
        auto_list = []
        tractor_list = []

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[4])
            c = class_list[d]
            if "car" in c:
                car_list.append([x1, y1, x2, y2])
            elif "bus" in c:
                bus_list.append([x1, y1, x2, y2])
            elif "bike" in c:
                bike_list.append([x1, y1, x2, y2])
            elif "LCV" in c:
                lcv_list.append([x1, y1, x2, y2])
            elif "MAV" in c:
                mav_list.append([x1, y1, x2, y2])
            elif "auto" in c:
                auto_list.append([x1, y1, x2, y2])
            elif "tractor" in c:
                tractor_list.append([x1, y1, x2, y2])

        total_list = (
            car_list
            + bus_list
            + bike_list
            + lcv_list
            + mav_list
            + auto_list
            + tractor_list
        )

        bbox_id = ntracker.update(total_list)

        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox

            results = cv2.pointPolygonTest(np.array(area, np.int32), (x4, y4), False)
            if results >= 0:
                vh[id] = (x3, y3)
                cv2.circle(frame, (x3, y3), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
            if id in vh:
                if counter.count(id) == 0:
                    counter.append(id)
                    bbox = [x3, y3, x4, y4]

                    if bbox in car_list:
                        car_counter.append(id)

                    elif bbox in bus_list:
                        bus_counter.append(id)

                    elif bbox in mav_list:
                        length = abs(y4 - y3)
                        if length < 120:
                            lcv_counter.append(id)
                            melcv_counter = mav_count_screenshot(
                                "/home/mic-710aix/ATCC/TP2_RHS/Actual/LCV_Types/LCV/Screenshots",
                                melcv_counter,
                                frame,
                                id,
                                "melcv",
                            )

                        else:
                            mav_counter.append(id)
                            if length > 130 and length <= 170:
                                axle_2_counter = mav_count_screenshot(
                                    "MAV/Axle_2/Screenshots",
                                    axle_2_counter,
                                    frame,
                                    id,
                                    "axle_2",
                                )
                            if length > 170 and length <= 220:
                                axle_3_counter = mav_count_screenshot(
                                    "MAV/Axle_3/Screenshots",
                                    axle_3_counter,
                                    frame,
                                    id,
                                    "axle_3",
                                )
                            elif length > 220 and length <= 250:
                                axle_4_counter = mav_count_screenshot(
                                    "MAV/Axle_4/Screenshots",
                                    axle_4_counter,
                                    frame,
                                    id,
                                    "axle_4",
                                )
                            elif length > 250 and length <= 280:
                                axle_5_counter = mav_count_screenshot(
                                    "MAV/Axle_5/Screenshots",
                                    axle_5_counter,
                                    frame,
                                    id,
                                    "axle_5",
                                )
                            elif length > 280 and length <= 300:
                                axle_6_counter = mav_count_screenshot(
                                    "MAV/Axle_6/Screenshots",
                                    axle_6_counter,
                                    frame,
                                    id,
                                    "axle_6",
                                )
                            elif length > 300 and length <= 360:
                                axle_7_plus_counter = mav_count_screenshot(
                                    "MAV/Axle_7_plus/Screenshots",
                                    axle_7_plus_counter,
                                    frame,
                                    id,
                                    "axle_7_plus",
                                )

                    elif bbox in auto_list:
                        auto_counter.append(id)

                    elif bbox in lcv_list:
                        lcv_counter.append(id)
                        minilcv_counter = mav_count_screenshot(
                            "LCV_Types/MINILCV/Screenshots",
                            minilcv_counter,
                            frame,
                            id,
                            "minilcv",
                        )

                    elif bbox in tractor_list:
                        tractor_counter.append(id)

                    elif bbox in bike_list:
                        bike_counter.append(id)

            cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 255), 2)

            car_count = len(car_counter)
            mav_count = len(mav_counter)
            lcv_count = len(lcv_counter)
            bus_count = len(bus_counter)
            auto_count = len(auto_counter)
            tractor_count = len(tractor_counter)
            bike_count = len(bike_counter)
            axle_2_count = len(axle_2_counter)
            axle_3_count = len(axle_3_counter)
            axle_4_count = len(axle_4_counter)
            axle_5_count = len(axle_5_counter)
            axle_6_count = len(axle_6_counter)
            axle_7_plus_count = len(axle_7_plus_counter)
            total_count = len(counter)

            # ----------------------------------------------------------------------------------------------------------

            if time.time() - start_time >= 60:
                # append data every 60 secs
                try:
                    append_data_to_csv(
                        "x.x.x.x",
                        "x.x.x.x",
                        "Area_1",
                        str(datetime.now().strftime("%d-%m-%Y %H:%M:%S")),
                        "x.xxx",
                        "x.xx",
                        total_count,
                        car_count,
                        axle_2_count,
                        mav_count,
                        bike_count,
                        bus_count,
                        lcv_count,
                        auto_count,
                        tractor_count,
                        axle_3_count,
                        axle_4_count,
                        axle_5_count,
                        axle_6_count,
                        axle_7_plus_count,
                        "xx",
                        "xx",
                        "xxxx",
                    )
                    msg = "Successfully data is appended to csv file"
                    logging.info(msg)
                    # print(msg)

                except Exception as e:
                    msg = f"while appending data to csv we got an error : {e}"
                    logging.critical(msg)
                    # print(msg)

                # reset counts

                counter = []
                car_counter = []
                mav_counter = []
                lcv_counter = []
                bus_counter = []
                auto_counter = []
                tractor_counter = []
                bike_counter = []
                axle_2_counter = []
                axle_3_counter = []
                axle_4_counter = []
                axle_5_counter = []
                axle_6_counter = []
                axle_7_plus_counter = []

                start_time = time.time()

        fps = (fps + (1.0 / (time.time() - t1))) / 2
        print("Inference FPS", fps)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    current_date = datetime.now()
    filename = f'script_log_{current_date.strftime("%Y%m%d%H%M%S")}.log'

    configure_logging(filename)
    main()
