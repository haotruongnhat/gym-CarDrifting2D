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

    progress = session_collection[Fields.progress.value]


    print("Total distance: {:2f} m".format(session_collection[Fields.distance.value][-1]))
    return pos_x, -pos_y, pos_z, progress

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

def find_nearest_idx(value, array):
    idx = np.abs(array - value).argmin()
    return idx

if __name__ == "__main__":
    l_pos_x, l_pos_y, l_pos_z, l_progress = get_track("map_data\\GR_Argolis_Fourketa Kourva_Left_Drive.npz")
    r_pos_x, r_pos_y, r_pos_z, r_progress = get_track("map_data\\GR_Argolis_Fourketa Kourva_Right_Drive.npz")

    ## Pacenotes
    pacenotes = []
    with open("map_data\\GR_Argolis_Fourketa Kourva_pacenotes.txt", "r") as f:
        commands = f.read().strip().split("\n")
        pacenotes = [(float(c.strip().split(" ")[0]), c.strip().split(" ")[1]) for c in commands[:-1]]

    sample_rate = 10
    skip_idx = 0

    l_min_x = np.min(l_pos_x)
    l_min_y = np.min(l_pos_y)
    r_min_x = np.min(r_pos_x)
    r_min_y = np.min(r_pos_y)

    min_x = np.min([l_min_x, r_min_x])
    min_y = np.min([l_min_y, r_min_y])

    scaling_factor = 0.5

    visualized_l_pos_x = (l_pos_x[skip_idx::sample_rate] - min_x)*scaling_factor
    visualized_l_pos_y = (l_pos_y[skip_idx::sample_rate] - min_y)*scaling_factor
    visualized_r_pos_x = (r_pos_x[skip_idx::sample_rate] - min_x)*scaling_factor
    visualized_r_pos_y = (r_pos_y[skip_idx::sample_rate] - min_y)*scaling_factor
    visualized_l_progress = l_progress[skip_idx::sample_rate]
    visualized_r_progress = r_progress[skip_idx::sample_rate]

    plt.plot(visualized_l_pos_x, visualized_l_pos_y, c='k')
    plt.plot(visualized_r_pos_x, visualized_r_pos_y, c='k')

    for prog, command in pacenotes:
        l_idx = find_nearest_idx(prog, visualized_l_progress)
        r_idx = find_nearest_idx(prog, visualized_r_progress)

        plt.plot([visualized_l_pos_x[l_idx], visualized_r_pos_x[r_idx]], [visualized_l_pos_y[l_idx], visualized_r_pos_y[r_idx]], c="green", linewidth=2)

    plt.show()


