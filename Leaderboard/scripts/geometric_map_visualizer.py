# Copyright (c) # Copyright (c) 2024 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a GeomtricmapCreator implementation.
"""

import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import argparse

# These should match the ones created by the simulation!
ROCK_UNCOMPLETED_VALUE = np.NINF
MAP_UNCOMPLETED_VALUE = np.NINF

UNCOMPLETED_COLOR = (0.2, 0.2, 0.2, 1)      # Dark gray
ROCK_GROUND_COLOR = (0.9, 0.9, 0.9)         # White
ROCK_COLOR = (0.55, 0.55, 0.55)             # Gray
ROCK_FALSE_P = (0.8, 0.25, 0.25)            # Red
ROCK_CORRECT = (0.25, 0.8, 0.25)            # Green
ROCK_FALSE_N = (0.25, 0.25, 0.8)            # Blue


def load_geometric_map(ground_path, agent_path):
    """Load the binary ground maps"""
    return (np.load(ground_path, allow_pickle=True), np.load(agent_path, allow_pickle=True))


def set_pretty_plot(axis, title, x_pos_range, x_data_range):
    """Changes the plot format for a prettier one"""
    axis.set_title(title, fontsize=20)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.xaxis.set_ticks_position('bottom')
    axis.yaxis.set_ticks_position('left')
    axis.xaxis.set_label_position('bottom')
    axis.yaxis.set_label_position('left')
    axis.set_aspect('equal', 'box')

    plt.sca(axis)
    plt.xticks(x_pos_range, x_data_range)
    plt.yticks(x_pos_range, x_data_range)


def create_height_map(data, axis, x_pos_range, x_data_range):
    """Creates the agent's height map"""
    color_map = cm.gist_earth
    color_map.set_bad(UNCOMPLETED_COLOR)

    pcm = axis.pcolormesh(data, cmap=color_map)
    plt.colorbar(pcm, ax=axis)

    set_pretty_plot(axis, "Height map", x_pos_range, x_data_range)


def create_height_error_map(data, axis, x_pos_range, x_data_range):
    """Creates the agent's height error map"""
    color_map = cm.terrain
    color_map.set_bad(UNCOMPLETED_COLOR)

    pcm = axis.pcolormesh(data, cmap=color_map)
    plt.colorbar(pcm, ax=axis)

    set_pretty_plot(axis, "Height error map", x_pos_range, x_data_range)


def create_rock_map(data, axis, x_pos_range, x_data_range):
    """Creates the agent's rock map"""
    colors = [ROCK_GROUND_COLOR, ROCK_COLOR]
    color_map = LinearSegmentedColormap.from_list('rock', colors, N=2)
    color_map.set_bad(UNCOMPLETED_COLOR)

    pcm = axis.pcolormesh(data, cmap=color_map)
    pcm.set_clim(-0.5, 1.5)  # Fixes the colors even if some 'sections' are missing
    legend_handles = [
        mpatches.Patch(color=ROCK_GROUND_COLOR, label='Ground'),
        mpatches.Patch(color=ROCK_COLOR, label='Rock'),
        mpatches.Patch(color=UNCOMPLETED_COLOR, label='Uncompleted')
    ]
    axis.legend(handles=legend_handles, bbox_to_anchor=(1, 0.5), loc='center left', borderaxespad=2)

    set_pretty_plot(axis, "Rock map", x_pos_range, x_data_range)


def create_rock_error_map(data, axis, x_pos_range, x_data_range):
    """Creates the agent's rock error map"""
    colors = [ROCK_FALSE_P, ROCK_CORRECT, ROCK_FALSE_N]
    color_map = LinearSegmentedColormap.from_list('rock', colors, N=3)
    color_map.set_bad(UNCOMPLETED_COLOR)

    pcm = axis.pcolormesh(data, cmap=color_map)
    pcm.set_clim(-1.5, 1.5)  # Fixes the colors even if some 'sections' are missing
    legend_handles = [
        mpatches.Patch(color=ROCK_CORRECT, label='Correct'),
        mpatches.Patch(color=ROCK_FALSE_P, label='False positive'),
        mpatches.Patch(color=ROCK_FALSE_N, label='False negative'),
        mpatches.Patch(color=UNCOMPLETED_COLOR, label='Uncompleted')
    ]
    axis.legend(handles=legend_handles, bbox_to_anchor=(1, 0.5), loc='center left', borderaxespad=2)

    set_pretty_plot(axis, "Rock error map", x_pos_range, x_data_range)


def get_axis_ranges(g_map):
    """Gets the axis ranges to match the map coordinates"""
    x_data_range = [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12]
    x_pos_range = []

    min_value = g_map[0,0,0]
    step_value =  g_map[1,0,0] - g_map[0,0,0]
    for value in x_data_range:
        # This position isn't a perfect match and might differ by 1
        position = int((value - min_value) / step_value) 
        x_pos_range.append(position)

    return (x_pos_range, x_data_range)


def visualize_geometric_map(g_map, a_map):
    """Visualize the maps using matplotlib as a 2 by 2 grid"""
    a_map[a_map == MAP_UNCOMPLETED_VALUE] = np.nan    # Detected as 'bad' with 'colormap.set_bad()'
    a_map[a_map == ROCK_UNCOMPLETED_VALUE] = np.nan   # Detected as 'bad' with 'colormap.set_bad()'

    # Get the data and rotate it so that x is horizontal and y vertical
    g_height_data = np.flip(np.rot90(g_map[:,:,2], 3), 1)
    g_rock_data = np.flip(np.rot90(g_map[:,:,3], 3), 1)
    a_height_data = np.flip(np.rot90(a_map[:,:,2], 3), 1)
    a_rock_data = np.flip(np.rot90(a_map[:,:,3], 3), 1)

    error_height_data = g_height_data - a_height_data
    error_rock_data = g_rock_data - a_rock_data
    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    fig.suptitle("Agent's height map data", fontsize=35)
    x_pos_range, x_data_range = get_axis_ranges(g_map)
    create_height_map(a_height_data, axs[0, 0], x_pos_range, x_data_range)
    create_height_error_map(error_height_data, axs[1, 0], x_pos_range, x_data_range)
    create_rock_map(a_rock_data, axs[0, 1], x_pos_range, x_data_range)
    create_rock_error_map(error_rock_data, axs[1, 1], x_pos_range, x_data_range)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig('ground_truth_comparison.png')
    plt.show()

    # Colormaps link: https://www.analyticsvidhya.com/blog/2020/09/colormaps-matplotlib/


def main():

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-gm', '--ground-map', required=True, help="Path to the .dat file with the ground truth")
    argparser.add_argument('-am', '--agent-map', required=True, help="Path to the .dat file with the agent's calculated groudn truth")
    args = argparser.parse_args()

    ground_map, agent_map = load_geometric_map(args.ground_map, args.agent_map)
    visualize_geometric_map(ground_map, agent_map)


if __name__ == '__main__':
    main()
