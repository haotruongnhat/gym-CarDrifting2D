from enum import Enum
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    npz_file = np.load(file_path)

    required_values = ['samples', 'game', 'save_version']
    for v in required_values:
        required_value_exists = v in npz_file
        if not required_value_exists:
            raise ValueError('Saved race doesn\'t contain the required field "{}"'.format(v))
    game_name = npz_file['game']
    save_version = npz_file['save_version']
    if save_version == '1.0.0':
        samples = npz_file['samples']
    else:
        raise ValueError('Unknown save version "{}" for game "{}"'.format(save_version, game_name))
    return samples

def convert_coordinate_system_3d(x, y, z):
    """
    Switch right-hand to left-hand coordinate system and vice versa.
    :param x: float scalar or numpy array
    :param y: float scalar or numpy array
    :param z: float scalar or numpy array
    :return:
    """

    return x, -z, y

def get_track(path):
    session_collection = load_data(path)
    pos_x, pos_y, pos_z = convert_coordinate_system_3d(
        session_collection[Fields.pos_x.value],
        session_collection[Fields.pos_y.value],
        session_collection[Fields.pos_z.value]
    )

    print("Total distance: {:2f} m".format(session_collection[Fields.distance.value][-1]))
    return pos_x, pos_y, pos_z

class Fields(Enum):
    run_time =            0
    lap_time =            1
    distance =            2
    progress =            3
    pos_x =               4
    pos_y =               5
    pos_z =               6
    speed_ms =            7
    vel_x =               8
    vel_y =               9
    vel_z =               10
    roll_x =              11
    roll_y =              12
    roll_z =              13
    pitch_x =             14
    pitch_y =             15
    pitch_z =             16
    susp_rl =             17
    susp_rr =             18
    susp_fl =             19
    susp_fr =             20
    susp_vel_rl =         21
    susp_vel_rr =         22
    susp_vel_fl =         23
    susp_vel_fr =         24
    wsp_rl =              25
    wsp_rr =              26
    wsp_fl =              27
    wsp_fr =              28
    throttle =            29
    steering =            30
    brakes =              31
    clutch =              32
    gear =                33
    g_force_lat =         34
    g_force_lon =         35
    current_lap =         36
    rpm =                 37  # / 10
    sli_pro_support =     38  # ignored
    car_pos =             39
    kers_level =          40  # ignored
    kers_max_level =      41  # ignored
    drs =                 42  # ignored
    traction_control =    43  # ignored
    anti_lock_brakes =    44  # ignored
    fuel_in_tank =        45  # ignored
    fuel_capacity =       46  # ignored
    in_pit =              47  # ignored
    sector =              48
    sector_1_time =       49
    sector_2_time =       50
    brakes_temp_rl =      51
    brakes_temp_rr =      52
    brakes_temp_fl =      53
    brakes_temp_fr =      54
    tyre_pressure_rl =    55  # ignored
    tyre_pressure_rr =    56  # ignored
    tyre_pressure_fl =    57  # ignored
    tyre_pressure_fr =    58  # ignored
    laps_completed =      59
    total_laps =          60
    track_length =        61
    last_lap_time =       62
    max_rpm =             63  # / 10
    idle_rpm =            64  # / 10
    max_gears =           65

if __name__ == "__main__":
    l_pos_x, l_pos_y, l_pos_z = get_track("map_data\\GR, Argolis, Fourketa Kourva - 562.1s - Left Drive.npz")
    r_pos_x, r_pos_y, r_pos_z = get_track("map_data\\GR, Argolis, Fourketa Kourva - 507.3s - Right Drive.npz")

    sample_rate = 10

    plt.plot(l_pos_x[::sample_rate], l_pos_y[::sample_rate], c='k')
    plt.plot(r_pos_x[::sample_rate], r_pos_y[::sample_rate], c='k')
    plt.show()