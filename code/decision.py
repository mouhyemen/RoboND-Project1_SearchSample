import numpy as np
import time

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

def robotBrake(Rover, brake):
    Rover.throttle = 0
    Rover.brake = brake
    Rover.steer = 0

def robotSteer(Rover, steer):
    if Rover.yaw > Rover.des_yaw:
        return -steer
    else:
        return steer

def robotThrottle(Rover, max_vel, throttle=0.5, brake=0, steer=0):
    Rover.max_vel = max_vel
    if abs(Rover.vel) < abs(Rover.max_vel):
        Rover.throttle = throttle
    else:
        Rover.throttle = 0
    Rover.brake = brake
    Rover.steer  = steer

def printMode(Rover, note = '', change=False):
            if change is True:
                print ("<<< {} MODE ----------------------------> {} MODE >>>".format(note.upper(), Rover.mode.upper()))
                return
            print ("---> {} mode, Vel: [{:.1f}], Yaw: [{:.1f}], dYaw: [{:.1f}]".format(
                    Rover.mode.upper(), Rover.vel, Rover.yaw, Rover.yaw-Rover.des_yaw))

def printData(sample,vel, dist, dyaw, mode):
    if sample:
        print ("Rock:[X], Vel:[{:.1f}], Dist:[{:.1f}], dYaw:[{:.1f}], Mode:[{}]".format(
            vel, dist, dyaw, mode))
    else:
        print ("Rock:[--], Vel:[{:.1f}], Dist:[{:.1f}], dYaw:[{:.1f}], Mode:[{}]".format(
            vel, dist, dyaw, mode))

def decision_step(Rover):
        Rover.counter += 1
        dx = Rover.pos[0] - Rover.rockx
        dy = Rover.pos[1] - Rover.rocky
        dist = np.sqrt(dx**2 + dy**2)
        delta_yaw = Rover.yaw - Rover.des_yaw
        yawCheck = -1.5 <= delta_yaw <= 1.5
        uncollected = Rover.samples_located - Rover.samples_collected
        frequency   = 10

        if Rover.nav_angles is not None:
                # Print Data and/or Motion-Mode
            if Rover.counter%frequency==0:
                    if Rover.mode == 'orient' or Rover.mode == 'pickup':
                        printData(Rover.near_sample, Rover.vel, dist, delta_yaw, Rover.mode)
    #                            elif Rover.mode is 'forward':
       #                             printMode(Rover)

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

            # Go near the rock
            elif Rover.mode == 'pickup':
                        chk1 = len(Rover.nav_angles) >= (Rover.stop_forward/16)  #close to the border
                        chk2 = len(Rover.nav_angles) >= (Rover.stop_forward/8)  #further from border
                        chk3 = len(Rover.nav_angles) >= (Rover.stop_forward/2)  #even further
                        # if chk1: # even if little area ahead
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

                        # else:   #if not enough ground ahead, go back
                        #             if Rover.counter%(frequency/2)==0:
                        #                     print (">>> TOO CLOSE {}. GO BACK!".format(len(Rover.nav_angles)))
                        #             robotThrottle(Rover, -1, throttle=-0.5, steer = 0)
                        #             Rover.mode = 'orient'

            # Navigate the terrain. If not enough ground ahead, change to 'stop' mode and brake
            if Rover.mode == 'forward':
                        if Rover.samples_located > 0 and uncollected is 0:
                            Rover.orient_flag = 0
                        steer = np.clip(np.mean(Rover.nav_angles*180 / np.pi), -15, 15)
                        # steer = steerAverage(Rover.steer_list, steer_old, size=3)

                        if len(Rover.nav_angles) >= Rover.stop_forward:
                            robotThrottle(Rover, 2, 1.5, steer = steer)
                        elif len(Rover.nav_angles) < Rover.stop_forward:
                            robotBrake(Rover, 15)
                            Rover.mode = 'stop'
                            printMode(Rover, 'forward', change=True)

            # Steer the robot. Then change to 'forward' mode
            elif Rover.mode == 'stop':
                        if Rover.vel <= 0.2:
                                    if len(Rover.nav_angles) < Rover.go_forward:
                                        Rover.throttle  = 0
                                        Rover.brake     = 0
                                        Rover.steer     = -15  # Could be more clever here
                                    elif len(Rover.nav_angles) >= Rover.go_forward:
                                        steer = np.clip(np.mean(Rover.nav_angles*180 / np.pi), -15, 15)
                                        robotThrottle(Rover, 5, 3, steer = steer)
                                        Rover.mode = 'forward'
                                        printMode(Rover, 'stop', change=True)
        else:
            Rover.throttle = 0
            Rover.steer = 0
            Rover.brake = 0

        return Rover
