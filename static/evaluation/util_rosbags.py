import numpy as np
from PIL import Image
import os

from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr
from rosbags.typesys import get_types_from_msg, register_types



def parse_msg(topic_name, time, data_array, msg, i, timestamp, start_time):

    time[i] = (timestamp - start_time)*1e-9     # nanoseconds to seconds

    if topic_name == "/twist_controller/command":
        data_array[i,0] = msg.linear.x
        data_array[i,1] = msg.linear.y
        data_array[i,2] = msg.linear.z
        data_array[i,3] = msg.angular.x
        data_array[i,4] = msg.angular.y
        data_array[i,5] = msg.angular.z
    
    if topic_name == "/fanuc/pose_velocity_controller/target":
        data_array[i,0] = msg.twist.linear.x
        data_array[i,1] = msg.twist.linear.y
        data_array[i,2] = msg.twist.linear.z
        data_array[i,3] = msg.twist.angular.x
        data_array[i,4] = msg.twist.angular.y
        data_array[i,5] = msg.twist.angular.z
        
    if topic_name == "/robot_pose" or topic_name == "/fanuc/pose/current":
        data_array[i,0] = msg.pose.position.x
        data_array[i,1] = msg.pose.position.y
        data_array[i,2] = msg.pose.position.z
        data_array[i,3] = msg.pose.orientation.x
        data_array[i,4] = msg.pose.orientation.y
        data_array[i,5] = msg.pose.orientation.z
        data_array[i,6] = msg.pose.orientation.w
    
    if topic_name == "/TrajectoryFromStates/DirectionOrientation":
        # [direction_x, direction_y, orientation, bbox_x, bbox_y]
        data_array[i, 0] =  msg.direction[0]
        data_array[i, 1] =  msg.direction[1]
        data_array[i, 2] =  msg.orientation
        data_array[i, 3] =  msg.bboxcenter[0]
        data_array[i, 4] =  msg.bboxcenter[1]

    if topic_name == "/TrajectoryFromStates/DirectionOrientationFiltered":
        # [direction_x, direction_y, orientation] 
        data_array[i, 0] =  msg.direction[0]
        data_array[i, 1] =  msg.direction[1]
        data_array[i, 2] =  msg.orientation
    
    if topic_name == "/ftn_axia":
        data_array[i, 0] = msg.force.x
        data_array[i, 1] = msg.force.y
        data_array[i, 2] = msg.force.z
        data_array[i, 3] = msg.torque.x
        data_array[i, 4] = msg.torque.y
        data_array[i, 5] = msg.torque.z
    if topic_name == "/joint_states":
        data_array[i,0,:] = msg.position
        data_array[i,1,:] = msg.velocity
        data_array[i,2,:] = msg.effort

    return data_array, time

def create_array(topic_name, len_data):
    if topic_name == "/twist_controller/command" or topic_name == "/fanuc/pose_velocity_controller/target":
        data_array = np.zeros((len_data, 6))
    if topic_name == "/robot_pose" or topic_name == "/fanuc/pose/current":
        data_array = np.zeros((len_data, 7))
    if topic_name == "/TrajectoryFromStates/DirectionOrientation":
        data_array = np.zeros((len_data, 5))    # [direction_x, direction_y, orientation, bbox_x, bbox_y]
    if topic_name == "/TrajectoryFromStates/DirectionOrientationFiltered":    
        data_array = np.zeros((len_data, 3))    # [direction_x, direction_y, orientation]
    if topic_name == "/ftn_axia":
        data_array = np.zeros((len_data, 6))
    if topic_name == "/joint_states":
        data_array = np.zeros((len_data, 3, 6))
    return data_array

def bag_to_array(topic_name, path_to_bag):
    # create reader instance
    with Reader(path_to_bag) as reader:
 
        connections = [x for x in reader.connections if x.topic == topic_name]
        #len_data = connections[0].msgcount
        
        len_data = 0
        for conn in connections:
            len_data = len_data + conn.msgcount

        time = np.zeros((len_data))
        start_time = reader.start_time
        data_array = create_array(topic_name=topic_name, len_data=len_data)
    
        for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=connections)):
            msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)

            data_array, time = parse_msg(topic_name=topic_name,
                                            time = time,
                                            data_array=data_array,
                                            msg=msg, 
                                            i=i, 
                                            timestamp=timestamp, 
                                            start_time=start_time)
    
    return data_array, time

def bag_to_rgb(path_to_bag, every_nth = 1, topic_name="/camera/color/image_raw"):
    # path_to_bag = r"src\baseline\measurements_with_depth\Rotor\Rotor_2025-03-21-14-40-34.bag"
    # topic_name = "/camera/color/image_raw" # /camera/depth/image_rect_raw
    save_path = os.path.join(os.sep.join(path_to_bag.split(os.sep)[:-1]), "images_video")
    os.makedirs(save_path, exist_ok=True)

    #process_bag(path_to_bag)

    with Reader(path_to_bag) as reader:
 
        connections = [x for x in reader.connections if x.topic == topic_name]
        for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=connections)):
            if i % every_nth == 0:
                msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                pil_image = Image.fromarray(image)
                path = os.path.join(save_path, "img_" + str(i).zfill(5) + ".png")
                while os.path.exists(path):
                    i += 1
                    path = os.path.join(save_path, "img_" + str(i).zfill(5) + ".png")    
                pil_image.save(path)

def bag_to_depth(path_to_bag, every_nth = 1, topic_name="/camera/depth/image_rect_raw"):
    save_path = os.path.join(os.sep.join(path_to_bag.split(os.sep)[:-1]), "depth_video")
    os.makedirs(save_path, exist_ok=True)

    with Reader(path_to_bag) as reader:
 
        connections = [x for x in reader.connections if x.topic == topic_name]
        for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=connections)):
            if i % every_nth == 0:
                msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                image = np.concatenate([image, image[:,:,[0]]], axis=2)
                image[:,:,1] = image[:,:,0]
                # image[:,:,2] = image[:,:,0]
                pil_image = Image.fromarray(image)
                path = os.path.join(save_path, "depth_" + str(i).zfill(5) + ".png")
                while os.path.exists(path):
                    i += 1
                    path = os.path.join(save_path, "depth_" + str(i).zfill(5) + ".png")    
                pil_image.save(path)


