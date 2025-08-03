if __name__ == "__main__":
    # Import necessary modules
    import csv
    import sys
    import cv2
    import math
    import os
    import random
    from sense_hat import SenseHat
    from orbit import ISS
    from exif import Image
    from datetime import datetime
    from picamera import PiCamera
    from time import sleep, perf_counter
    from pathlib import Path
    from logzero import logger, logfile

    sense = SenseHat()
    acceleration_records = []
    Integrated_velocity_records = []

    def get_time(image):
        with open(image, 'rb') as image_file:
            img = Image(image_file)
            time_str = img.get("datetime_original")
            time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
        return time

    def get_time_difference(photo_1, photo_2):
        time_1 = get_time(photo_1)
        time_2 = get_time(photo_2)
        time_difference = time_2 - time_1
        return time_difference.seconds

    def convert_to_cv(image_path_1, image_path_2):
        image_1_cv = cv2.imread(str(image_path_1), 0)
        image_2_cv = cv2.imread(str(image_path_2), 0)
        return image_1_cv, image_2_cv

    def calculate_features(image_1_cv, image_2_cv, feature_number):
        orb = cv2.ORB_create(nfeatures=feature_number)
        keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
        keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
        return keypoints_1, keypoints_2, descriptors_1, descriptors_2

    def calculate_matches(descriptors_1, descriptors_2):
        brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = brute_force.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
        match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
        resize = cv2.resize(match_img, (1600, 600), interpolation=cv2.INTER_AREA)
        cv2.imshow('matches', resize)
        cv2.waitKey(0)
        cv2.destroyWindow('matches')

    def find_matching_coordinates(keypoints_1, keypoints_2, matches):
        coordinates_1 = []
        coordinates_2 = []
        for match in matches:
            image_1_idx = match.queryIdx
            image_2_idx = match.trainIdx
            (x1, y1) = keypoints_1[image_1_idx].pt
            (x2, y2) = keypoints_2[image_2_idx].pt
            coordinates_1.append((x1, y1))
            coordinates_2.append((x2, y2))
        return coordinates_1, coordinates_2

    def calculate_mean_distance(coordinates_1, coordinates_2):
        all_distances = 0
        merged_coordinates = list(zip(coordinates_1, coordinates_2))
        for coordinate in merged_coordinates:
            x_difference = coordinate[0][0] - coordinate[1][0]
            y_difference = coordinate[0][1] - coordinate[1][1]
            distance = math.hypot(x_difference, y_difference)
            all_distances += distance
        return all_distances / len(merged_coordinates)

    def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
        distance = feature_distance * GSD / 100000.0
        speed = distance / time_difference
        return speed
#-----------------------------------------------------------------

    def integrate_acceleration_trapezoidal(acceleration_values, time_interval):
        velocity = [0.0] * len(acceleration_values[0])
        for i in range(1, len(acceleration_values)):
            for j in range(len(acceleration_values[i])):
                km_per_s2 = acceleration_values[i][j] / 1000  # Convert m/s^2 to km/s^2
                velocity[j] += (km_per_s2 + acceleration_values[i-1][j]) / 2 * time_interval
        return velocity

    def calculate_average_linear_speed(velocity_values, time_interval):
        displacement = sum(velocity_values) * time_interval
        average_speed = displacement / time_interval
        return average_speed
    
    def kalman_filter(measurement):
        # Kalman filter parameters
        Q = 0.0001  # Process noise covariance
        R = 0.1  # Measurement noise covariance
        x = 0  # Initial state
        P = 1  # Initial covariance

        # Kalman filter update
        K = P / (P + R)
        x = x + K * (measurement - x)
        P = (1 - K) * P + Q

        return x
    # Create an instance of PiCamera
    camera = PiCamera()
    #Rocco 2-12-2023
    #best framerate to get a shorter time duration
    #camera.framerate = 120
    # Set capture mode to photo
    #camera.capture_mode = 'photo'
    #camera.exposure_mode = 'auto'  # Set exposure mode
    #camera.awb_mode = 'auto'  # Set auto white balance
    # Set image processing parameters
    #shutter_speed = camera.exposure_speed # Adjust shutter speed (in microseconds)
    #camera.shutter_speed = shutter_speed
    #camera.iso = 100  # Adjust the ISO
    # Set image resolution to the highest supported resolution
    #camera.resolution = camera.MAX_RESOLUTION
    # Set a small delay between captures (in seconds)
    #capture_delay = 0.1
    #Let camera settle for a fixed time before capturing
    #sleep(2)
    #-------------------------------------------------------------------------------
    
    final_speed = 0
    #record_count = 0 #rocco 3-12-2023
    
    base_folder = Path(__file__).parent.resolve()
    try:
        time_differences = []
        
        for i in range(1, 17):
            #Obtain accelerometer records
            acceleration = sense.get_accelerometer_raw()
            filtered_acceleration = kalman_filter(acceleration["x"])
            camera.capture(f"{base_folder}/photo{i}.jpg")
            print("image%d download successfully",i)
            x = round(filtered_acceleration, 2)
            y = round(acceleration['y'], 2)
            z = round(acceleration['z'], 2)
            record = [x, y, z]
            acceleration_records.append(record)
            print(record)
            sleep(5)
            #record_count += 1 #rocco 3-12-2023
            #if record_count >= 10: #rocco 3-12-2023
                #break #rocco 3-12-2023
        
        total_speed = 0
        speed_list = []
        camera.close()
        
        #acceleration_records = acceleration_records[1:] #Rocco 10-12-2023
        # Calculate average acceleration
        average_acceleration = [sum(axis) / len(acceleration_records) for axis in zip(*acceleration_records)]
        print("Average acceleration (x, y, z):", average_acceleration)

        for index in range(1, 15):
            photo_1 = f"{base_folder}/photo{index}.jpg"
            photo_2 = f"{base_folder}/photo{index + 1}.jpg"
            time_difference = get_time_difference(photo_1, photo_2)
            print("hello")
            time_differences.append(time_difference)

        #time_differences = time_differences[1:]  #Rocco 2-12-2023
        
        """print("===========================")
        print(time_differences[0])
        time_differences[0] = random.choice(time_differences[1:])
        print("posterio ===========================")"""
        print(time_differences[0])
        
        for index in range(1, 15):
            photo_1 = f"{base_folder}/photo{index}.jpg"
            photo_2 = f"{base_folder}/photo{index + 1}.jpg"
            

            image_1_cv, image_2_cv = convert_to_cv(photo_1, photo_2)
            keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000)
            matches = calculate_matches(descriptors_1, descriptors_2)
            coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
            average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
            speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_differences[index-1])
            total_speed = total_speed + speed
            print("hahaha: ",time_difference)
        """for index in range(1, 10):
            photo_1 = f"{base_folder}/photo{index}.jpg"
            photo_2 = f"{base_folder}/photo{index + 1}.jpg"
            time_difference = get_time_difference(photo_1, photo_2)
            time_differences.append(time_difference)
            time_differences = time_differences[1:]  # Remove the first record      
            #--------------------------------
            image_1_cv, image_2_cv = convert_to_cv(photo_1,photo_2) # Create OpenCV image objects
            keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000) # Get keypoints and descriptors
            matches = calculate_matches(descriptors_1, descriptors_2) # Match descriptors
            coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
            average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
            speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference)
            total_speed = total_speed + speed""" #Rocco 2-12-2023

        print("Total speed:", total_speed)
        Average_Speed = total_speed/15.0
        print("Photo speed:", Average_Speed)
        
        time_interval = time_difference
        velocity = integrate_acceleration_trapezoidal(acceleration_records, time_interval)
        Integrated_velocity_records.append(velocity)
        average_speed = calculate_average_linear_speed(velocity, time_interval)
        estimate_kms_formatted = "{:.5f}".format((Average_Speed + average_speed)/2)
        output_speed = estimate_kms_formatted
        final_speed = estimate_kms_formatted

        with open(base_folder/"result.txt","w") as file:
            file.write(str(final_speed))

        print("Data written to", base_folder/"result.txt")
        print(final_speed)

        print(time_differences)
        average_time_difference = sum(time_differences) / len(time_differences)
        print("Average time difference (seconds):", average_time_difference)

    except Exception as e:
        print("An error occurred:", str(e))

    print("Integrated velocity (x, y, z):", velocity)
    print("Average linear speed:", average_speed)
    print("Final linear speed:", final_speed)
#delete no.1 time, acceleration value and photo data