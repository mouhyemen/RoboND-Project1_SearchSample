import numpy as np
import cv2

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

'''
The `detect_rocks()` function will detect rocks within a certain RGB threshold value.
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

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1] / 2).astype(np.float)
    return x_pixel, y_pixel

# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))  # keep same size as input image
    maskImg = np.ones_like(img[:, :, 0])
    height, width = int(img.shape[0]), int(img.shape[1])
    # mask = cv2.warpPerspective(maskImg, M, (img.shape[1], img.shape[0]))
    return warped

# Create obstacle map from navigable terrain map and the mask
def create_obs_map(nav_map):
    _,cnts,_ = cv2.findContours(nav_map.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    obs_map = np.zeros_like(nav_map)
    cv2.drawContours(obs_map, cnts, -1, (1), 15)
    white = nav_map[:,:] > 0
    obs_map[white] = 0
    return obs_map

def center_rock(rover_rockx, rover_rocky, world_rockx, world_rocky):
    rock_dist, rock_angles = to_polar_coords(rover_rockx, rover_rocky)
    min_idx = np.argmin(rock_dist)
    x_cen = world_rockx[min_idx]
    y_cen = world_rocky[min_idx]
    return x_cen, y_cen

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
            # Perform perception steps to update Rover()
            # NOTE: camera image is coming to you in Rover.img

            # 1) Define source and destination points for perspective transform
            roll = np.absolute(Rover.roll)
            pitch = np.absolute(Rover.pitch)

            dst_size = 5
            bottom_offset = 6
            image = Rover.img

            source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
            destination = np.float32([[image.shape[1] / 2 - dst_size, image.shape[0] - bottom_offset],
                                      [image.shape[1] / 2 + dst_size, image.shape[0] - bottom_offset],
                                      [image.shape[1] / 2 + dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                                      [image.shape[1] / 2 - dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                                      ])

            # 2) Apply perspective transform
            warped = perspect_transform(image, source, destination)

            # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
            threshed = color_thresh(warped)
            threshed = box_threshed(threshed)
            obs_map = create_obs_map(threshed)

            # 4) Update Rover.vision_image (this will be displayed on left side of screen)
            Rover.vision_image[:, :, 2] = threshed * 255  # nav terrain color-thresholded binary image
            Rover.vision_image[:, :, 0] = obs_map * 255  # obstacle color-thresholded binary image

            # 5) Convert map image pixel values to rover-centric coords
            xpix, ypix = rover_coords(threshed)        # array of rover_xy values for nav map
            obs_xpix, obs_ypix = rover_coords(obs_map)  # array of rover_xy values for obs map

            # 6) Convert rover-centric pixel values to world coordinates
            xpos = Rover.pos[0]  # current x position
            ypos = Rover.pos[1]  # current y position
            yaw = Rover.yaw    # current yaw angle
            scale = 10
            world_size = Rover.worldmap.shape[0]  # 200 x 200

            # navigable xy-world coordinates
            xworld, yworld = \
                pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale)

            # obstacle xy-world coordinates
            obs_xworld, obs_yworld = \
                pix_to_world(obs_xpix, obs_ypix, xpos, ypos, yaw, world_size, scale)

            # 7) Update Rover worldmap (to be displayed on right side of screen)
            Rover.worldmap[yworld, xworld, 2] += 20  # setting nav terrain to blue
            Rover.worldmap[obs_yworld, obs_xworld, 0] += 1  # setting obstacles to red

            # Detecting rocks in the map
            # rock_map = detect_rocks(warped)
            rock_map = detect_rocks(image)

            samples_uncollected = Rover.samples_located - Rover.samples_collected
            if samples_uncollected > 0:
                    if Rover.orient_flag is 0 and not Rover.picking_up:
                        Rover.mode = 'orient'
                        Rover.orient_flag = 1

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

                    Rover.des_yaw = np.arctan2(ypos-rock_ycen, xpos-rock_xcen) * 180/np.pi
                    if 0 <= abs(Rover.des_yaw) <= 180:
                          Rover.des_yaw = 180 + Rover.des_yaw

                    samples_uncollected = Rover.samples_located - Rover.samples_collected
                    if samples_uncollected > 0:
                                    if Rover.orient_flag is 0 and not Rover.picking_up:
                                        Rover.mode = 'orient'
                                        Rover.orient_flag = 1

            else:
                Rover.vision_image[:, :, 1] = 0  # if not found, mark it black

            # 8) Convert rover-centric pixel positions to polar coordinates
            dist, angles = to_polar_coords(xpix, ypix)

            # Update Rover pixel distances and angles
            Rover.nav_angles = angles

            return Rover


