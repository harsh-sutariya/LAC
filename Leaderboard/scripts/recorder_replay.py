# # Copyright (c) # Copyright (c) 2024 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import argparse
import threading
import sys
import os
import pygame
import datetime
import astropy
import lunarsky

import carla

FOLLOW = False
FOLLOW_DISTANCE = 3
FPS = 20
FIRST_FRAME = 0
LAST_FRAME = False
REPLAY_SPEED = 1
TIME = 0

WEATHER_SPEED = 60            # s


class astrobody():

    def __init__(self, name, datatime, moon_loc):
        
        self.name = name
        t = astropy.time.Time(datatime)
        obj = astropy.coordinates.get_body(name, t)
        obj = obj.transform_to(lunarsky.LunarTopo(location=moon_loc, obstime=datatime))
        self.obj = obj

    def __str__(self):
        return self.name +", "+ str(self.obj)


class Sun(object):

    def __init__(self, world, initial_date, moon_loc):
        self._world = world
        self._weather = world.get_weather()
        self._t = 0.0
        self.initial_date = initial_date
        self.moon_loc = moon_loc
        self.update(0)

    def update(self, delta_seconds):

        self._t += delta_seconds / 3600
        date = str(self.initial_date + datetime.timedelta(hours=self._t))

        with astropy.coordinates.solar_system_ephemeris.set('builtin'):
            sun = astrobody("sun", date, self.moon_loc)

        self._weather.sun_azimuth_angle = float(sun.obj.az.value)
        self._weather.sun_altitude_angle = float(sun.obj.alt.value)
        self._world.set_weather(self._weather)


class Earth(object):

    UA2KM = 149597871
    SCALE_FACTOR = 0.1

    def __init__(self, world, initial_date, moon_loc):

        self.world = world
        self._t = 0.0
        self.initial_date = initial_date
        self.moon_loc = moon_loc
        self.earth_pos_scaled = self.load_and_get_location(initial_date)

        bp = world.get_blueprint_library().filter("*earth*")[0]
        self.earth = world.try_spawn_actor(bp, carla.Transform(carla.Location(x=self.earth_pos_scaled[0],y=self.earth_pos_scaled[1],z=self.earth_pos_scaled[2])))
        newrot = carla.Rotation(roll=180.)
        self.earth.set_transform(carla.Transform(location=self.earth.get_location(),rotation=newrot))
        self.update(0)

    def load_and_get_location(self, date):

        with astropy.coordinates.solar_system_ephemeris.set('builtin'):
            earth = astrobody("earth", date, self.moon_loc)

        earth.obj = earth.obj.transform_to(lunarsky.MCMF(obstime=date))
        earth_pos = [earth.obj.x.value*self.UA2KM, earth.obj.y.value*self.UA2KM, earth.obj.z.value*self.UA2KM]
        earth_pos_scaled = [coord*self.SCALE_FACTOR for coord in earth_pos]

        return earth_pos_scaled

    def update(self, delta_seconds):

        self._t += delta_seconds / 3600
        date = str(self.initial_date + datetime.timedelta(hours=self._t))

        self.earth_pos_scaled = self.load_and_get_location(date)

        newloc = carla.Location(x=self.earth_pos_scaled[0],y=self.earth_pos_scaled[1],z=self.earth_pos_scaled[2])
        newrot = carla.Rotation(yaw=self._t/24.*360.,roll=180)
        self.earth.set_transform(carla.Transform(newloc,newrot))

    def destroy(self):
        self.earth.destroy()


def tick(world):
    global TIME, REPLAY_SPEED
    world.tick()
    TIME += world.get_snapshot().delta_seconds * REPLAY_SPEED


def recorder_utilities(client):
    global LAST_FRAME, REPLAY_SPEED, FOLLOW
    stop = False

    while not stop and not LAST_FRAME:
        data = input("\nInput the next action: ")
        try:
            int_data = float(data)
            print("  Setting the replayer factor to {}".format(int_data))
            client.set_replayer_time_factor(int_data)
            REPLAY_SPEED = int_data
        except ValueError:
            if data == "R":
                print("  Time: {}".format(round(TIME, 3)))
            elif data == "S":
                stop = True
            elif data == "F":
                if FOLLOW:
                    print("  Setting the camera to follow the IPEx")
                else: 
                    print("  Stopping the camera from following the IPEx")
                FOLLOW = not FOLLOW
            else:
                print("\033[93mIgnoring unknown command '{}'\033[0m".format(data))

    LAST_FRAME = True


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', metavar='H', default='localhost', help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument('--port', metavar='P', default=2000, type=int, help='TCP port of CARLA Simulator (default: 2000)')

    # Recorder arguments
    argparser.add_argument('-f', '--file', default='', required=True, help='File to be executed')
    argparser.add_argument('--start-time', default=0, type=float, help='Start time of the recorder')
    argparser.add_argument('--end-time', default=0, type=float, help='End time of the recorder')
    argparser.add_argument('--factor', default=1, type=float, help='Initial recorder factor')
    argparser.add_argument('--static-sky', action="store_true", help='Disables the Sun and Earth movement')

    args = argparser.parse_args()

    global TIME, LAST_FRAME, REPLAY_SPEED, FOLLOW

    TIME = args.start_time
    REPLAY_SPEED = args.factor

    client = None
    world = None
    sun = None
    earth = None
    ipex = None
    lander = None
    rocks = None

    if not os.path.exists(args.file):
        print("WARNING: The specified '.log' file does not exist. Shutting down")
        sys.exit(-1)

    # Get the client
    print("\n\033[1m> Setting the simulation\033[0m")
    client = carla.Client(args.host, args.port)
    client.set_timeout(200.0)
    file_info = client.show_recorder_file_info(args.file, False)

    # Synchronous mode provides a smoother motion of the camera
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1/FPS
    world.apply_settings(settings)

    # Get the duration of the recorder (only if the end time is 0, aka until the recorder end)
    duration = args.end_time
    if not duration:
        duration = float(file_info.split("Duration: ")[-1].split(" ")[0])

    print("\033[1m> Starting the replayer\033[0m")
    client.replay_file(args.file, args.start_time, args.end_time, 0)
    client.set_replayer_time_factor(args.factor)

    # Weather
    if not args.static_sky:
        initial_date = datetime.datetime.strptime('2023-01-15 00:00:00', "%Y-%m-%d %H:%M:%S")
        moon_loc = lunarsky.MoonLocation(lat=-90,lon=0)
        sun = Sun(world, initial_date, moon_loc)
        earth = Earth(world, initial_date, moon_loc)

    tick(world)
    ipex = world.get_actors().filter('*ipex*')[0]
    lander = world.get_actors().filter('*lander*')[0]
    rocks = world.get_actors().filter('*rock*')

    try:
        print("\033[1m> Running the recorder utility. Use:")
        print(" - R: to record the replayer timestamp data")
        print(" - S: to stop the script")
        print(" - F: to start/stop following the IPEx")
        print(" - A number: to change the speed's factor of the replayer\033[0m")
        print("\nNote that this script has to be manually shut down as it won't end when the replayer stops")

        t1 = threading.Thread(target=recorder_utilities, args=(client, ))
        t1.start()

        spectator = world.get_spectator()

        clock = pygame.time.Clock()
        while not LAST_FRAME:
            clock.tick_busy_loop(20)
            tick(world)
            if TIME >= duration:
                LAST_FRAME = True

            if FOLLOW:
                ipex_transform = ipex.get_transform()
                ipex_vec = ipex_transform.get_forward_vector()
                spectator_location = ipex_transform.location + carla.Location(x=-ipex_vec.x*FOLLOW_DISTANCE,
                                                                              y=-ipex_vec.y*FOLLOW_DISTANCE,
                                                                              z=2)
                spectator_rotation = carla.Rotation(yaw=ipex_transform.rotation.yaw, pitch=-25)
                spectator_transform = carla.Transform(spectator_location, spectator_rotation)
                spectator.set_transform(spectator_transform)

            if not args.static_sky:
                sun.update(WEATHER_SPEED/FPS)
                earth.update(WEATHER_SPEED/FPS)

        client.stop_replayer(False)

    except KeyboardInterrupt:
        pass
    finally:

        if ipex is not None:
            ipex.destroy()
        if lander is not None:
            lander.destroy()
        if rocks is not None:
            for rock in rocks:
                rock.destroy()

        if earth is not None:
            earth.destroy()

        if client is not None:
            client.set_replayer_time_factor(1)

        if world is not None:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)


if __name__ == '__main__':
    main()
