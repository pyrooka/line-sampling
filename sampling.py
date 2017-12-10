#! /usr/bin/env python3

"""
Simple line sampling script.
"""

import sys
import argparse

import numpy as np
from PIL import Image
from matplotlib import pyplot


def get_args():
    """
    Get the arguments.

    Returns:
        (image, line length, dimension, line thickness) if OK else None.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Input image with white background and \
                                                 a drawed line with one pixel size')
    parser.add_argument('linelength', type=int, help='Line length on the image in milimeter (>0).')
    parser.add_argument('dimension', type=int, help='Grid dimension in milimeter. (>0).')
    parser.add_argument('--thickness', type=int, help='Line thickness (default 1).')
    args = parser.parse_args()

    if args.image and args.linelength and args.dimension:
        return (args.image, args.linelength, args.dimension, args.thickness)

    return


def read_image(img):
    """
    Read/load the image.

    Args:
        img: Path to the image.

    Returns:
        Inverted numpy array (with 1s and 0s).
    """

    try:
        img_file = Image.open(img)

        img_bw = img_file.convert('1')

        img_array = np.array(img_bw, dtype=np.uint8)

        # Invert. 0 -> 1, 1 -> 0
        return 1 - img_array

    except Exception:
        # Ooops.
        raise


def pixel_per_mm(img, length, thickness=1):
    """
    Calculate the number of the pixels in one milimeter.

    Args:
        img: Binary image in numpy array.
        length: Line length in milimeter.
        thickness: Line thickness in pixel.

    Returns:
        Line length in milimeter.
    """

    # The length of the line is the sum of pixels with value 1 divided by the thickness
    lenght_pixel = np.sum(img) / thickness

    return lenght_pixel / length


def create_grid(img, ppm, dimension):
    """
    Create the grid with the given resolution.
    Starts from the top-left corner.

    Args:
        img: numpy array for the shape.
        ppm: pixel per milimeter.
        dimension: resolution of the grid.

    Returns:
        tuple: (numpy array with the grid, list of the grid points)
    """

    # Grid resolution in pixel
    resolution_pixel = int(round(dimension * ppm))

    grid = np.zeros(img.shape, dtype=np.uint8)

    # Where the grid lines are fill with 1s.
    # Rows.
    for i in range(0, grid.shape[0]):
        if i % resolution_pixel == 0:
            grid[i] = 1

    # Columns.
    grid = grid.T
    for i in range(0, grid.shape[0]):
        if i % resolution_pixel == 0:
            grid[i] = 1

    grid = grid.T

    grid_points = []
    # Number of the grid points.
    x_points_count = grid.shape[1] // resolution_pixel
    y_points_count = grid.shape[0] // resolution_pixel

    # Calculate the coordinates.
    for i in range(0, x_points_count + 1):
        for j in range(0, y_points_count + 1):
            grid_points.append((j * resolution_pixel, i * resolution_pixel))

    # Add the coordinates of the last columns too.
    for i in range(0, y_points_count + 1):
        grid_points.append((i * resolution_pixel, grid.shape[1] - 1))

    return (grid, grid_points)


def get_section_points(img, grid):
    """
    Find all the sections of the image and grid.

    Args:
        img: numpy array.
        grid: numpy array.

    Returns:
        List of points.
    """

    # List with the coordinates of line points.
    y_coord, x_coord = np.where(img == 1)
    line = [coord for coord in zip(y_coord, x_coord)]

    # Section point is where the grid and the image value are 1 too.
    section_points = []
    for coord in line:
        if grid[coord] == 1:
            section_points.append(coord)

    # Last colum sections if exists.
    last_column = img[:, -1:]
    end = np.where(last_column == 1)

    if len(end[0]) and len(end[1]):
        section_points.append((end[0][0], img.shape[1] - 1))

    return section_points


def find_nodes(grid_points, section_points):
    """
    Find the nearest grid points from the seciton points. Called nodes.

    Args:
        grid_points: list of the grid points.
        section_points: list of the section points.

    Returns:
        List of the nodes.
    """

    grid_points = np.array(grid_points)
    section_points = np.array(section_points)

    # Make uniq.
    nodes = set()

    for section_point in section_points:
        point_diff = np.sum(abs(grid_points - section_point), axis=1)
        min_point_coord = np.argmin(point_diff)
        nodes.add(tuple(grid_points[min_point_coord]))

    # Sort by X the Y.
    nodes = sorted(nodes, key=lambda node: (node[1], node[0]))

    return nodes


def calculate_linear_length(nodes, dimension, ppm):
    """
    Calculate the length of the line on the grid.

    Args:
        nodes: sorted nodes.
        dimension: grid resolution in milimeter.
        ppm: pixel per milimeter.

    Return:
        The line length in milimeter.
    """

    resolution_pixel = dimension * ppm

    lines_count = 0

    nodes = np.array(nodes)

    # Calculate the diffs between points.
    for i in range(0, len(nodes) - 1):
        diff = np.absolute(nodes[i+1] - nodes[i])

        # Y diff.
        lines_count += round(diff[0] / resolution_pixel)
        # X diff.
        lines_count += round(diff[1] / resolution_pixel)


    return int(dimension * lines_count)


def create_linear_line(grid, nodes):
    """
    Continous line on the grid which following the drawed line.

    Args:
        grid: numpy array.
        nodes: sorted nodes.

    Returns:
        numpy array.
    """

    line = np.full(grid.shape, 255, dtype=np.uint8)
    nodes = np.array(nodes)

    for i in range(0, len(nodes) - 1):
        diff = nodes[i+1] - nodes[i]

        # X.
        for j in range(0, abs(diff[1]) + 1):
            if diff[1] < 0:
                j *= -1
            y_coord, x_coord = nodes[i]
            line[y_coord, x_coord+j] = 0

        # Y.
        for j in range(0, abs(diff[0]) + 1):
            if diff[0] < 0:
                j *= -1
            y_coord, x_coord = nodes[i]
            line[y_coord+j, x_coord+diff[1]] = 0

    return line


def plot(img, grid, line, original_length, linear_length):
    """
    Plot the result.

    Args:
        img: numpy array.
        grid: numpy array.
        line: numpy array.
        original_length: length of the original line in milimeter.
        linear_length: lenght of the calculated linear (grid) line in milimeter.

    Returns:
        -
    """

    grid_plot = (1 - grid) * 255
    img_plot = (1 - img) * 255

    img_plot = np.dstack((np.dstack((img_plot, img_plot)), img_plot))

    img_plot[grid_plot == 0] = (150, 150, 150)
    img_plot[line == 0] = (255, 0, 0)

    # Display the result.
    pyplot.title('Original lenght: {} mm\nLinear lenght: {} mm'.format(original_length,
                                                                       linear_length))
    pyplot.imshow(img_plot)
    pyplot.show()


def main():
    """
    Main.
    """

    args = get_args()

    if not args:
        sys.exit('Not enough arguments.')

    img, length, dimension, thickness = args

    try:
        matrix = read_image(img)
    except Exception as ex:
        sys.exit('Error occured while reading the image.' + str(ex))

    # Calculate the pixel per milimeter.
    if thickness:
        ppm = pixel_per_mm(matrix, length, thickness)
    else:
        ppm = pixel_per_mm(matrix, length)

    # Create the grid.
    grid, grid_points = create_grid(matrix, ppm, dimension)
    if not len(grid_points):
        sys.exit('Not enough grid points.')

    section_points = get_section_points(matrix, grid)
    if not len(section_points):
        sys.exit('Not enough section points.')

    nodes = find_nodes(grid_points, section_points)
    if not len(nodes):
        sys.exit('Not enough nodes.')

    linear_length = calculate_linear_length(nodes, dimension, ppm)

    line = create_linear_line(grid, nodes)

    plot(matrix, grid, line, length, linear_length)


if __name__ == '__main__':
    main()
