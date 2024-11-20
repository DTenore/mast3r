import numpy as np

def write_prediction(filename, prediction):
    with open(filename, "a") as file:
        file.write(prediction + "\n")

def load_intrinsics(filepath):
    intrinsics_dict = {}
    frame_width = None
    frame_height = None

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 7:
                continue
            
            frame_path = parts[0]
            fx, fy, cx, cy = map(float, parts[1:5])
            width, height = map(int, parts[5:7])

            # Set frame width and height once
            if frame_width is None and frame_height is None:
                frame_width = width
                frame_height = height

            # Create the 3x3 intrinsic matrix
            intrinsics_matrix = np.array([
                [fx, 0,  cx],
                [0,  fy, cy],
                [0,  0,  1]
            ])
            
            # Store the matrix in the dictionary
            intrinsics_dict[frame_path] = intrinsics_matrix

    return intrinsics_dict, frame_width, frame_height
   