import cv2
import numpy as np
import math

def calculate_optimal_focal_length(image_width, image_height, rotation_angle):
    # Convert angle to radians
    angle_rad = math.radians(abs(rotation_angle))

    # Get the maximum dimension of the image
    max_dimension = max(image_width, image_height)

    # Base focal length calculation
    base_focal_length = max_dimension * (1.0 + math.sin(angle_rad))

    # Round to nearest 100 for cleaner values
    return int(round(base_focal_length / 100.0) * 100)

def crop(input_img):
    # Find the bounding box of non-white pixels
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return input_img

    # Combine all contours to get bounding rect
    all_points = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_points)

    # Add some padding
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(input_img.shape[1] - x, w + padding * 2)
    h = min(input_img.shape[0] - y, h + padding * 2)

    return input_img[y:y+h, x:x+w]

def resize_to_original_aspect_ratio(input_img, target_width, target_height):
    # Calculate scaling factor to fit within target dimensions while preserving aspect ratio
    scale_width = target_width / input_img.shape[1]
    scale_height = target_height / input_img.shape[0]
    scale = min(scale_width, scale_height)

    # Calculate new dimensions
    new_width = int(input_img.shape[1] * scale)
    new_height = int(input_img.shape[0] * scale)

    # Resize image
    resized = cv2.resize(input_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create canvas of target size with white background
    canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255

    # Calculate position to paste the resized image (centered)
    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2

    # Copy the resized image to the canvas
    canvas[y:y+new_height, x:x+new_width] = resized

    return canvas

def main():
    # Read image
    input_path = "/Users/akrylov/PycharmProjects/pfm/backgrounds/imagedopNew_4.png"
    img = cv2.imread(input_path)

    if img is None:
        print("Error: Could not read the input image.")
        return

    # Original dimensions
    orig_height, orig_width = img.shape[:2]

    # Set rotation angle
    rotation_angle = 45.0  # Y-axis rotation in degrees

    # Define custom rotation axis point (relative to image center)
    # Values are in percentage of width/height from the center
    # (0,0) is the center of the image
    # (-0.5,-0.5) is the top-left corner
    # (0.5,0.5) is the bottom-right corner
    axis_x_percent = 0#-0.5  # 50% to the left of center
    axis_y_percent = 0  # at center

    # Convert to pixel coordinates relative to center
    axis_x = axis_x_percent * orig_width
    axis_y = axis_y_percent * orig_height

    # Calculate the optimal focal length
    focal_length = calculate_optimal_focal_length(orig_width, orig_height, rotation_angle)
    print(f"Calculated optimal focal length: {focal_length}")

    # Calculate a larger output size to ensure no clipping
    output_scale = 2.0 + abs(rotation_angle) / 45.0
    output_width = int(orig_width * output_scale)
    output_height = int(orig_height * output_scale)

    # 2D to 3D projection with custom axis
    proj_2d_to_3d = np.array([
        [1, 0, -(orig_width / 2.0 + axis_x)],
        [0, 1, -(orig_height / 2.0 + axis_y)],
        [0, 0, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # Initialize rotation matrices
    rx = np.eye(4, dtype=np.float32)
    ry = np.eye(4, dtype=np.float32)
    rz = np.eye(4, dtype=np.float32)

    # Translation matrix - using calculated focal length
    trans = np.eye(4, dtype=np.float32)
    trans[2, 3] = focal_length

    # 3D to 2D projection with centered output
    proj_3d_to_2d = np.array([
        [focal_length, 0, output_width / 2.0, 0],
        [0, focal_length, output_height / 2.0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)

    # Set rotation angles
    x = -60
    y = 0#rotation_angle
    z = 0

    # Convert angles to radians
    ax = x * (math.pi / 180.0)
    ay = y * (math.pi / 180.0)
    az = z * (math.pi / 180.0)

    # Set up rotation matrices
    # X-axis rotation
    rx[1, 1] = math.cos(ax)
    rx[1, 2] = -math.sin(ax)
    rx[2, 1] = math.sin(ax)
    rx[2, 2] = math.cos(ax)

    # Y-axis rotation
    ry[0, 0] = math.cos(ay)
    ry[0, 2] = -math.sin(ay)
    ry[2, 0] = math.sin(ay)
    ry[2, 2] = math.cos(ay)

    # Z-axis rotation
    rz[0, 0] = math.cos(az)
    rz[0, 1] = -math.sin(az)
    rz[1, 0] = math.sin(az)
    rz[1, 1] = math.cos(az)

    # Combine rotations: r = rx @ ry @ rz
    temp = rx @ ry
    r = temp @ rz

    # Create transformation matrix: final = proj_3d_to_2d @ trans @ r @ proj_2d_to_3d
    step1 = r @ proj_2d_to_3d
    step2 = trans @ step1
    final_transform = proj_3d_to_2d @ step2

    # Apply transformation with larger output dimensions
    rotated = cv2.warpPerspective(
        img, final_transform, (output_width, output_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )

    # Crop the rotated image to remove excess white space
    cropped = crop(rotated)

    # Resize back to original dimensions (or maintain aspect ratio within original dimensions)
    result = resize_to_original_aspect_ratio(cropped, orig_width, orig_height)

    # Save the output
    output_path = "./output/image-rotated-custom-axis.png"
    cv2.imwrite(output_path, result)

    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    main()