
# Project 1: Search and Sample Return
---


## 1. Project Summary

### 1.1 Objectives:
* Map atleast 40% of the map
* Atleast 60% fidelity
* Locate at least one rock sample

### 1.2 Outcomes:
* Mapped 98% of the map
* With 70% fidelity
* Located and picked up all 6 samples

## 2. Detection of navigable terrain, obstacles, and rock samples

### 2.1 Navigable Terrain
The image coming in from the Rover's perspective is first warped using OpenCV's perspective transform function. Then a color thresholding filter is applied on the warped image. The result is as follows.
![threshed](/images/threshed.png)

To improve the fidelity, the thresholded figure shown above is not taken as is. Instead, a mask is applied on it to extract only a certain portion of the thresholded-warped figure. The result is shown below.

![box_nav](/images/box_nav.png)


```python
'''
Identify pixels above the threshold
Threshold of RGB > 160 does a nice job of identifying ground pixels only
'''
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:, :, 0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:, :, 0] > rgb_thresh[0]) \
        & (img[:, :, 1] > rgb_thresh[1]) \
        & (img[:, :, 2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

'''
The idea is to limit the amount of navigable terrain to increase fidelity
Do a series of erosions and dilations to remove any sharp edges or blobs.
Create a rectangular mask that comes out of the robot's POV.
Use the mask on the dilated (thresholded) image
'''
def box_threshed(threshed):
    eroded   = cv2.erode(threshed, None, iterations=2)
    dilated  = cv2.dilate(eroded, None, iterations=4)

    mask_box = np.zeros_like(threshed)
    boxHeight, boxWidth = mask_box.shape[0], mask_box.shape[1]
    up       = int(boxHeight/2+20)
    down  = int(boxHeight)
    left     = int(boxWidth/2-80)
    right   = int(boxWidth/2+80)
    mask_box[up:down, left:right] = 1

    final   = cv2.bitwise_and(dilated, dilated, mask = mask_box)
    return final
```

### 2.2 Obstacle outlining
The method employed here is to simply outline the navigable terrain detected using contour detection. First, a series of erosions and dilations are performed to remove any edges and blobs. Then contours are detected and outlined.
![outline_obstacle](/images/outline_obstacle.png)


```python
'''
Create obstacle map from navigable terrain map.
First detect the contours and draw it.
Then mask out the interior of the contour.
'''
def create_obs_map(nav_map):
    _,cnts,_ = cv2.findContours(nav_map.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    obs_map = np.zeros_like(nav_map)
    cv2.drawContours(obs_map, cnts, -1, (1), 15)
    white = nav_map[:,:] > 0
    obs_map[white] = 0
    return obs_map
```

### 2.3 Rocks detection
A simple color filtering is applied on RGB color space to detect the rock samples. The result is shown below.
![detect_rock](/images/detect_rock.png)


```python
'''
This function will detect rocks within a certain RGB threshold value.
Yellow rocks have a high red and green value with a low blue value.
'''
def detect_rocks(img, rock_thresh=(110, 110, 60)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:, :, 0])

    above_thresh = (img[:, :, 0] > rock_thresh[0]) \
        & (img[:, :, 1] > rock_thresh[1]) \
        & (img[:, :, 2] < rock_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select
```

## 3. Perception

The `perception_step(Rover)` carries out a series of steps and updates the Rover object. The high-level breakdown of `perception_step(Rover)` is the following:

### 3.1 Perspective Transform, Navigable Terrain and Obstacle Mapping
First, we create our source and destination points that will be used to perform a perspective transform. The warped image is color filtered using `color_thresh()` and masked into a smaller region using the `box_threshed()` function. The `create_obs_map()` function outlines the obstacle region around the navigable region. The resultant is shown below.


```python
# 1) Define source and destination points for perspective transform
dst_size, bottom_offset = 5, 6
source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
destination = np.float32([[image.shape[1] / 2 - dst_size, image.shape[0] - bottom_offset],
                          [image.shape[1] / 2 + dst_size, image.shape[0] - bottom_offset],
                          [image.shape[1] / 2 + dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                          [image.shape[1] / 2 - dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                          ])

# 2) Apply perspective transform
warped, mask = perspect_transform(image, source, destination)

# 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
threshed = color_thresh(warped)
threshed = box_threshed(threshed)
obs_map  = create_obs_map(threshed)
```

![step1_image](/images/step1_image.png)

### 3.2 Updating rover vision image and coordinate transformations

Now that we have information on navigable terrain and obstacle region, we update the `Rover.vision_image` attribute. We then convert the pixels of both navigable and obstacle terrains to rover-centric coordinates using `rover_coords()` function. These coordinates are then converted to world coordinates using `pix_to_world()` function. Now that we have information on world centric coordinates, we update the `Rover.worldmap` attribute with navigable and obstacle pixels.


```python
# 4) Update Rover.vision_image (this will be displayed on left side of screen)
Rover.vision_image[:, :, 2] = threshed * 255  # nav terrain color-thresholded binary image
Rover.vision_image[:, :, 0] = obs_map * 255   # obstacle color-thresholded binary image

# 5) Convert map-image pixel values to rover-centric coords
xpix, ypix = rover_coords(threshed)         # array of rover_xy values for nav map
obs_xpix, obs_ypix = rover_coords(obs_map)  # array of rover_xy values for obs map

# 6) Convert rover-centric pixel values to world coordinates
xpos, ypos = Rover.pos[0], Rover.pos[1]  # current x & y position
yaw        = Rover.yaw      # current yaw angle
scale      = 10
world_size = Rover.worldmap.shape[0]  # 200 x 200

xworld, yworld = pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale) # navigable xy-world coordinates

obs_xworld, obs_yworld = \
pix_to_world(obs_xpix, obs_ypix, xpos, ypos, yaw, world_size, scale)# obstacle xy-world coordinates

# 7) Update Rover worldmap (to be displayed on right side of screen)
Rover.worldmap[yworld, xworld, 2] += 20  # setting nav terrain to blue
Rover.worldmap[obs_yworld, obs_xworld, 0] += 1  # setting obstacles to red
```

### 3.3 Rock detection

Next, we move onto detecting rocks from the Rover's camera's point of view using the `detect_rocks()` function explained above. After that, we check if a rock sample is located but not picked up. If that is the case, then the `Rover.mode` attribute changes to `'orient'` mode.

#### Note: `Rover.mode` attributes are explained in more detail under Section #4: Autonomous Navigation.


```python
# 8) Detecting rocks in the map
rock_map = detect_rocks(image)

# 9) Checking if any rock samples are not collected. 
# If True, then the Rover goes to 'orient' mode
samples_uncollected = Rover.samples_located - Rover.samples_collected
if samples_uncollected > 0:
        if Rover.orient_flag is 0 and not Rover.picking_up:
            Rover.mode = 'orient'
            Rover.orient_flag = 1
```

The following segment of code checks if a rock is indeed detected. If so, then first we need to convert the rock-pixels into rover-centric coordinates and then to world-centric coordinates. `center_rock()` function is used for finding the center of the rock and then displaying it on the worldmap.

A desired yaw angle is calculated using the inverse arc2 tangent function. Since the worldmap is inverted along y-axis, a mapping of the desired yaw angle is being computed.


```python
def center_rock(rover_rockx, rover_rocky, world_rockx, world_rocky):
    rock_dist, rock_angles = to_polar_coords(rover_rockx, rover_rocky)
    min_idx = np.argmin(rock_dist)
    x_cen = world_rockx[min_idx]
    y_cen = world_rocky[min_idx]
    return x_cen, y_cen

if rock_map.sum() > 0:
    rock_xpix, rock_ypix = rover_coords(rock_map)

    rock_xworld, rock_yworld = \
        pix_to_world(rock_xpix, rock_ypix, xpos, ypos, yaw, world_size, scale)

    rock_xcen, rock_ycen = center_rock(rock_xpix, rock_ypix, rock_xworld, rock_yworld)

    Rover.worldmap[rock_ycen, rock_xcen, 1] = 255
    Rover.vision_image[:, :, 1] = rock_map * 255  # rock sample color-thresholded binary img

    Rover.rockx = rock_xcen
    Rover.rocky = rock_ycen
    Rover.rock_angle = np.arctan2(rock_ycen, rock_xcen) * 180/np.pi
    
    # 11) Calculating the desired yaw angle to orient towards the rock
    Rover.des_yaw = np.arctan2(ypos-rock_ycen, xpos-rock_xcen) * 180/np.pi
    if 0 <= abs(Rover.des_yaw) <= 180:
        Rover.des_yaw = 180 + Rover.des_yaw

    # 12) Checking if any rock samples are not collected. 
    # If True, then the Rover goes to 'orient' mode
    samples_uncollected = Rover.samples_located - Rover.samples_collected
    if samples_uncollected > 0:
        if Rover.orient_flag is 0 and not Rover.picking_up:
            Rover.mode = 'orient'
            Rover.orient_flag = 1

else:
    Rover.vision_image[:, :, 1] = 0  # if not found, mark it black
    
```

## 4. Autonomous Navigation

The `decision_step(Rover)` function in the `decision.py` script is used for navigating the Rover. 

### 4.1 Wrapper Functions for Rover navigation
Certain wrapper functions are created for easier deployment of throttle, brakes, steering, and setting maximum velocities.


```python
# An average moving filter of steering angles
def steerAverage(steer_list, steer, size=5):
    if np.isnan(steer):
        steer = 0
    steer_list.append(steer)
    if len(steer_list) > size:
        steer_mean = sum(steer_list)/len(steer_list)
        steer_list = steer_list.pop(0)      # remove first element
        return steer_mean
    else:
        return steer

# Applying brakes with a certain brake value
def robotBrake(Rover, brake):
    Rover.throttle = 0
    Rover.brake = brake
    Rover.steer = 0

# Steering cw or ccw based on desired yaw angle
def robotSteer(Rover, steer):
    if Rover.yaw > Rover.des_yaw:
        return -steer
    else:
        return steer

# Throttling forward based on certain parameters
def robotThrottle(Rover, max_vel, throttle=0.5, brake=0, steer=0):
    Rover.max_vel = max_vel
    if abs(Rover.vel) < abs(Rover.max_vel):
        Rover.throttle = throttle
    else:
        Rover.throttle = 0
    Rover.brake = brake
    Rover.steer  = steer
```

### 4.2 Picking up rock sample

A conditional is checked on every update if the Rover is near a rock sample or not. If it is, then it should pick up the rock sample and change `Rover.mode` to `forward` mode.


```python
#check if the sample is around. if so, then pick it up.
if Rover.near_sample:
    if Rover.vel > 0.5:     # If moving, then brake.
        robotBrake(Rover, 5)
        print (">>> SAMPLE IS NEAR.".format(Rover.brake))
    Rover.throttle, Rover.steer = 0, 0

    if not Rover.picking_up:    # Pick up sample. Go to forward mode
        print (">>> Picking up rock sample #{}".format(Rover.samples_located))
        Rover.send_pickup = True
        Rover.max_vel = 3
        Rover.mode = 'forward'
        Rover.des_yaw = 0
```

### 4.3 Types of `Rover.mode`

There are 4 modes that I am using for my Rover. They are: `orient, pickup, forward, stop`

* `Orient Mode`: If a rock sample is detected, the Rover should stop and rotate towards the rock sample. Based on how much the orientation angle is going, the steering angle is determined. The steering angles are 3, 10, or 15 units. Once oriented, `Rover.mode` attribute changes to `pickup` mode.


```python
# Orient to face the rock sample
if Rover.mode == 'orient':
    if yawCheck:    #if facing sample, go to 'pickup' mode
        Rover.mode = 'pickup'
        printMode(Rover, note = 'orient', change=True)
    else:   # If not facing sample, first stop, then orient
        if Rover.vel > 0.5:  # If moving, then brake
                robotBrake(Rover, 10)
        else:
            if abs(delta_yaw) >= 40:
                Rover.steer = robotSteer(Rover, 15)
            elif 10<= abs(delta_yaw) < 40:
                Rover.steer = robotSteer(Rover, 10)
            elif abs(delta_yaw) < 10:
                Rover.steer = robotSteer(Rover, 3)
            Rover.brake = 0
            Rover.throttle = 0
```

* `Pickup Mode`: Once the Rover faces the rock sample, it should throttle towards it while maintaining the desired yaw angle. The chk1, chk2, chk3 variables are simply flags that let the Rover know if an obstacle is near in descending order.


```python
# Go near the rock
elif Rover.mode == 'pickup':
    chk1 = len(Rover.nav_angles) >= (Rover.stop_forward/16)  #close to the border
    chk2 = len(Rover.nav_angles) >= (Rover.stop_forward/8)  #further from border
    chk3 = len(Rover.nav_angles) >= (Rover.stop_forward/2)  #even further

    if dist < 0.5 and chk3: # if within certain distance of rock, first stop, then steer
        print ("Very close to rock but plenty of room ahead -> {}".format(len(Rover.nav_angles)))
        if Rover.vel > 0.25:
            robotBrake(Rover, 5)
        else:
            robotThrottle(Rover, 1, throttle=0, steer = 15)
    elif dist < 0.8 and chk2:     # if close to rock and border near
        print ("Close to rock but little room ahead - > {}".format(len(Rover.nav_angles)))
        if Rover.vel > 0.25:
            robotBrake(Rover, 5)
        else:
            robotThrottle(Rover, 1, throttle=0, steer = 15)
    elif  -3 <= delta_yaw <= 3:  # if facing sample, drive straight to it
        print ("going straight")
        robotThrottle(Rover, 1.5, throttle=1)
    else:   # if not near and not facing sample, drive and steer towards it
        print ("steering")
        if delta_yaw >0:
            robotThrottle(Rover, 1, throttle=0.5, steer = -5)
        else:
            robotThrottle(Rover, 1, throttle=0.5, steer = 5)
```

* `Forward Mode`: Navigate the terrain based on the angle computed from average navigable vector. If there is no navigable terrain ahead, then change to 'stop' mode.


```python
if Rover.mode == 'forward':
    if Rover.samples_located > 0 and uncollected is 0:
        Rover.orient_flag = 0
        
    steer_old = np.clip(np.mean(Rover.nav_angles*180 / np.pi), -15, 15)
    steer = steerAverage(Rover.steer_list, steer_old, size=3)
    
    if len(Rover.nav_angles) >= Rover.stop_forward:
        robotThrottle(Rover, 4, 1.5, steer = steer)
    elif len(Rover.nav_angles) < Rover.stop_forward:
        robotBrake(Rover, 15)
        Rover.mode = 'stop'
        printMode(Rover, 'forward', change=True)
```

* `Stop Mode`: If there are no navigable pixels ahead, then steer clockwise until navigable pixels detected and change the mode back to `forward` mode.


```python
#check if the sample is around. if so, then pick it up.
if Rover.near_sample:
    if Rover.vel > 0.5:     # If moving, then brake.
        robotBrake(Rover, 5)
        print (">>> SAMPLE IS NEAR.".format(Rover.brake))
    Rover.throttle, Rover.steer = 0, 0

    if not Rover.picking_up:    # Pick up sample. Go to forward mode
        print (">>> Picking up rock sample #{}".format(Rover.samples_located))
        Rover.send_pickup = True
        Rover.max_vel = 3
        Rover.mode = 'forward'
        Rover.des_yaw = 0
```

## 5. Worldmap and Vision Image:

The vision_image and worldmap are shown on the left and right of the screen.

### 5.1 Vision image
If a rock sample is detected, it is shown in green color. The navigable terrain is colored in blue whereas the obstacle outlining this navigable terrain is marked in red. A sample image is shown below

![vision_rock](/images/vision_rock.png)

### 5.2 Worldmap image
The worldmap contains some statistics on the Rover such as time elapsed, percentage of the terrain mapped, and fidelity. Apart from stats, the navigable terrain, obstacles, and rocks are highlighted in the map.

I have added additional stats and labels to the worldmap.

* `Rover's location and orientation`: The position and orientation of the Rover is shown on the worldmap as well.


```python
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
```

* `Grids`: The ground truth of the navigable region is divided into equally spaced grids of 10x10 pixels each. This lets the Rover know it is currently in which grid.


```python
# worldmap divided into equally spaced grids
checkBlacks = map_add[:,:,:] < 10
size = 10
for i in range(0,199,size):
    cv2.line(map_add, (i, 0), (i, 199), (255,255,255), 1)
    cv2.line(map_add, (0, i), (199, i), (255,255,255), 1)
map_add[checkBlacks] = 0
grid_x, grid_y = getGrid(Rover.pos[0], Rover.pos[1], size)
```

A sample of the worldmap is shown below.

![worldmap](/images/worldmap.png)

## 6. Improvements

### 6.1 Increased fidelity
I would like to work on increasing the fidelity of my Rover. Currently, I am averaging around 70% after masking my warped navigable terrain. Prior to this, my fidelity would average around 55% only.

### 6.2 Returning to home position with all 6 samples
Currently, the Rover is only able to collect all 6 samples but I did not implement any logic for the Rover to return to its initial place. This is something I would like to work on.

### 6.3 Avoid re-visiting terrains
In order to tackle this, I had created grids in the worldmap for the Rover to localize itself into grid cells. As the Rover is entering a new grid, each grid gets updated with a weight. However, I have not implemented any algorithm after this for the Rover to unnecessarily visit certain terrains. An example of the grid generated is shown below. It is a 20x20 matrix.

![grid_map](/images/grid_map.png)
