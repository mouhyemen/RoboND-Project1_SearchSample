import numpy as np
import cv2
from PIL import Image
from io import BytesIO, StringIO
import base64
import time


# Define a function to convert telemetry strings to float independent of
# decimal convention
def convert_to_float(string_to_convert):
      if ',' in string_to_convert:
            float_value = np.float(string_to_convert.replace(',', '.'))
      else:
            float_value = np.float(string_to_convert)
      return float_value

def getGrid(x,y,size):
      for row in range(0, 199, size):
            for column in range(0, 199, size):
                  if row <= x < row + size and column <= y < column+size:
                        return [row, column]

def update_rover(Rover, data):
      # Initialize start time and sample positions
      if Rover.start_time == None:
            Rover.start_time = time.time()
            Rover.total_time = 0
            samples_xpos = np.int_([convert_to_float(pos.strip()) for pos in data["samples_x"].split(';')])
            samples_ypos = np.int_([convert_to_float(pos.strip()) for pos in data["samples_y"].split(';')])
            Rover.samples_pos = (samples_xpos, samples_ypos)
            Rover.samples_to_find = np.int(data["sample_count"])
      # Or just update elapsed time
      else:
            tot_time = time.time() - Rover.start_time
            if np.isfinite(tot_time):
                  Rover.total_time = tot_time
      # Print out the fields in the telemetry data dictionary
      # print(data.keys())
      # The current speed of the rover in m/s
      Rover.vel = convert_to_float(data["speed"])
      # The current position of the rover
      Rover.pos = [convert_to_float(pos.strip()) for pos in data["position"].split(';')]
      # The current yaw angle of the rover
      Rover.yaw = convert_to_float(data["yaw"])
      # The current yaw angle of the rover
      Rover.pitch = convert_to_float(data["pitch"])
      # The current yaw angle of the rover
      Rover.roll = convert_to_float(data["roll"])
      # The current throttle setting
      Rover.throttle = convert_to_float(data["throttle"])
      # The current steering angle
      Rover.steer = convert_to_float(data["steering_angle"])
      # Near sample flag
      Rover.near_sample = np.int(data["near_sample"])
      # Picking up flag
      Rover.picking_up = np.int(data["picking_up"])
      # Update number of rocks collected
      Rover.samples_collected = Rover.samples_to_find - \
          np.int(data["sample_count"])

      # print('speed =',Rover.vel, 'position =', Rover.pos, 'throttle =',
      # Rover.throttle, 'steer_angle =', Rover.steer, 'near_sample:', Rover.near_sample,
      # 'picking_up:', data["picking_up"], 'sending pickup:', Rover.send_pickup,
      # 'total time:', Rover.total_time, 'samples remaining:', data["sample_count"],
      # 'samples collected:', Rover.samples_collected)

      # print('speed =', Rover.vel, 'position =', Rover.pos, 'steer_angle =', Rover.steer,
      #       'yaw =', Rover.yaw, 'roll =', Rover.roll, 'pitch =', Rover.pitch)

      # Get the current image from the center camera of the rover
      imgString = data["image"]
      image = Image.open(BytesIO(base64.b64decode(imgString)))
      Rover.img = np.asarray(image)

      # Return updated Rover and separate image for optional saving
      return Rover, image

# Define a function to create display output given worldmap results
def create_output_images(Rover):

      # Create a scaled map for plotting and clean up obs/nav pixels a bit
      if np.max(Rover.worldmap[:, :, 2]) > 0:
            nav_pix = Rover.worldmap[:, :, 2] > 0
            navigable = Rover.worldmap[:, :, 2] * (255 / np.mean(Rover.worldmap[nav_pix, 2]))
      else:
            navigable = Rover.worldmap[:, :, 2]

      if np.max(Rover.worldmap[:, :, 0]) > 0:
            obs_pix = Rover.worldmap[:, :, 0] > 0
            obstacle = Rover.worldmap[:, :, 0] * (255 / np.mean(Rover.worldmap[obs_pix, 0]))
      else:
            obstacle = Rover.worldmap[:, :, 0]

      likely_nav = navigable >= obstacle
      obstacle[likely_nav] = 0
      plotmap = np.zeros_like(Rover.worldmap)
      plotmap[:, :, 0] = obstacle
      plotmap[:, :, 2] = navigable
      plotmap = plotmap.clip(0, 255)

      # Overlay obstacle and navigable terrain map with ground truth map
      map_add = cv2.addWeighted(plotmap, 1, Rover.ground_truth, 0.5, 0)

      # Check whether any rock detections are present in worldmap
      rock_world_pos = Rover.worldmap[:, :, 1].nonzero()
      # If there are, we'll step through the known sample positions
      # to confirm whether detections are real
      samples_located = 0
      test_rock_x = 0
      test_rock_y = 0
      if rock_world_pos[0].any():
            rock_size = 2
            for idx in range(len(Rover.samples_pos[0])):
                  test_rock_x = Rover.samples_pos[0][idx]
                  test_rock_y = Rover.samples_pos[1][idx]
                  rock_sample_dists = np.sqrt((test_rock_x - rock_world_pos[1])**2 +
                                              (test_rock_y - rock_world_pos[0])**2)
                  # If rocks were detected within 3 meters of known sample positions
                  # consider it a success and plot the location of the known
                  # sample on the map
                  if np.min(rock_sample_dists) < 10:
                        samples_located += 1
                        Rover.samples_located = samples_located
                        Rover.next_rock = 1
                        map_add[test_rock_y - rock_size:test_rock_y + rock_size,
                                test_rock_x - rock_size:test_rock_x + rock_size, :] = 255
                        # print ("World-map Rocks: x {}, y {}".format(test_rock_x, test_rock_y))


      # Calculate some statistics on the map results
      # First get the total number of pixels in the navigable terrain map
      tot_nav_pix = np.float(len((plotmap[:, :, 2].nonzero()[0])))
      # Next figure out how many of those correspond to ground truth pixels
      good_nav_pix = np.float(
          len(((plotmap[:, :, 2] > 0) & (Rover.ground_truth[:, :, 1] > 0)).nonzero()[0]))
      # Next find how many do not correspond to ground truth pixels
      bad_nav_pix = np.float(len(((plotmap[:, :, 2] > 0) & (Rover.ground_truth[:, :, 1] == 0)).nonzero()[0]))
      # Grab the total number of map pixels
      tot_map_pix = np.float(len((Rover.ground_truth[:, :, 1].nonzero()[0])))
      # Calculate the percentage of ground truth map that has been successfully
      # found
      perc_mapped = round(100 * good_nav_pix / tot_map_pix, 1)
      # Calculate the number of good map pixel detections divided by total pixels
      # found to be navigable terrain
      if tot_nav_pix > 0:
            fidelity = round(100 * good_nav_pix / (tot_nav_pix), 1)
      else:
            fidelity = 0
      # Flip the map for plotting so that the y-axis points upward in the
      # display
      # map_add = np.flipud(map_add).astype(np.float32)



      # Plotting rover's location on the worldmap
      rov_x = int (Rover.pos[0])
      rov_y = int (Rover.pos[1])
      rover_size = 3
      map_add[rov_y - rover_size:rov_y + rover_size, \
                  rov_x - rover_size:rov_x + rover_size, 0] = 255

      # Drawing an arrow coming out of the rover
      length = 30
      arrow_x =  int (rov_x + length * np.cos(Rover.yaw * np.pi / 180.0))
      arrow_y =  int (rov_y + length * np.sin(Rover.yaw * np.pi / 180.0))
      cv2.line(map_add, (rov_x+2,rov_y+2), (arrow_x,arrow_y), (255,0,0), 2)

      # worldmap divided into equally spaced grids
      checkBlacks = map_add[:,:,:] < 10
      size = 10
      for i in range(0,199,size):
            cv2.line(map_add, (i, 0), (i, 199), (255,255,255), 1)
            cv2.line(map_add, (0, i), (199, i), (255,255,255), 1)
      map_add[checkBlacks] = 0
      grid_x, grid_y = getGrid(Rover.pos[0], Rover.pos[1], size)

      if grid_x is not Rover.grid_x or grid_y is not Rover.grid_y:
            Rover.grid_x, Rover.grid_y = grid_x, grid_y
            grid_xy = [grid_x, grid_y]
            if grid_xy not in Rover.grid_list:
                  Rover.grid_list.append(grid_xy)
                  Rover.weight_map[int(grid_y/size), int(grid_x/size)] = Rover.weight
                  Rover.weight += 1
                  # print (Rover.weight_map, '\n')

      # Add some text about map and rock sample detection results
      offset = 110
      # cv2.putText(map_add, "Yaw: " + str(np.round(Rover.yaw, 1)) + ' deg',       (0, 20),
      #             cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
      cv2.putText(map_add, "Grid: " + str(np.round(grid_x, 1)) +
             ", " + str(np.round(grid_y, 1)),                                                                  (0, 20),
                  cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
      cv2.putText(map_add, "Time: " + str(np.round(Rover.total_time, 1)) + ' s',(0, 10 + offset),
                  cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
      cv2.putText(map_add, "Mapped: " + str(perc_mapped) + '%',                     (0, 25 + offset),
                  cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
      cv2.putText(map_add, "Fidelity: " + str(fidelity) + '%',                                (0, 40 + offset),
                  cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
      cv2.putText(map_add, "Rocks: " + str(np.round(Rover.rockx, 1)) +
             ", " + str(np.round(Rover.rocky, 1)),                                                    (0, 55 + offset),
                  cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
      cv2.putText(map_add, "  Located: " + str(samples_located),                       (0, 70 + offset),
                  cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
      cv2.putText(map_add, "  Collected: " + str(Rover.samples_collected),         (0, 85 + offset),
                  cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
      # Convert map and vision image to base64 strings for sending to server
      pil_img = Image.fromarray(map_add.astype(np.uint8))
      buff = BytesIO()
      pil_img.save(buff, format="JPEG")
      encoded_string1 = base64.b64encode(buff.getvalue()).decode("utf-8")

      pil_img = Image.fromarray(Rover.vision_image.astype(np.uint8))
      buff = BytesIO()
      pil_img.save(buff, format="JPEG")
      encoded_string2 = base64.b64encode(buff.getvalue()).decode("utf-8")

      return encoded_string1, encoded_string2
