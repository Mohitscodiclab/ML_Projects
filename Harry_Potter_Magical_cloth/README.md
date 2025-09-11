

# Harry Potter Invisible Cloth Project

## Overview

This project brings the magic of Harry Potter to life! Using OpenCV and Python, I've created a real-time "Invisible Cloak" effect that makes objects of a specific color disappear by replacing them with the background. The program allows users to select their own cloak color and automatically captures the background for a seamless invisibility effect.

## Features

- **Interactive Color Selection**: Users can choose any colored cloth as their "invisible cloak"
- **Automatic Background Capture**: The program captures the background for 3 seconds at startup
- **Real-time Invisibility Effect**: Objects of the selected color become invisible in real-time
- **Noise Reduction**: Uses morphological transformations to clean up the detection
- **User-friendly Interface**: Clear visual instructions guide users through the setup process

## How It Works

1. **Background Capture**: The program first captures the background for 3 seconds.
2. **Color Selection**: The user positions their colored cloth in the center of the frame and presses 'c' to capture its color.
3. **Color Detection**: The program converts frames to HSV color space and creates a mask for the selected color.
4. **Noise Removal**: Morphological operations (opening and dilation) are applied to remove noise from the mask.
5. **Invisibility Effect**: The program replaces the detected color with the background using bitwise operations, creating the illusion of invisibility.

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- A webcam

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/mohitscodiclab/invisible-cloa.git
   cd invisible-cloak
   ```

2. Install the required packages:
   ```bash
   pip install opencv-python numpy
   ```

## Usage

1. Run the program:
   ```bash
   python invisible_cloak.py
   ```

2. **Background Capture Phase**:
   - Keep your camera still for 3 seconds while the program captures the background.
   - A countdown timer will show on screen.

3. **Color Selection Phase**:
   - Position your colored cloth in the center of the frame (inside the yellow rectangle).
   - Press 'c' to capture the color of the cloth.
   - The program will analyze the color and proceed to the next step.

4. **Invisibility Effect**:
   - The invisibility effect will start automatically.
   - Any objects with the selected color will become "invisible".
   - Press 'q' to quit the program.

## Customization

You can customize the following parameters in the code:

- **Color Tolerance**: Adjust the tolerance values in the `lower_bound` and `upper_bound` arrays to make the color detection more or less sensitive.
- **Morphological Operations**: Modify the kernel size and operations to better handle noise in your environment.
- **Background Capture Duration**: Change the 3-second background capture duration by modifying the `start_time` calculation.

## Known Issues and Limitations

- The effect works best with solid, saturated colors.
- Similar colors in the background might cause unintended invisibility.
- Lighting changes after background capture can affect the quality of the effect.
- The cloak must be a single, consistent color for best results.

## Future Improvements

- Implement dynamic background updating to handle camera movement
- Add support for multiple cloak colors
- Create a GUI for easier parameter adjustment
- Implement edge feathering for smoother blending
- Add the ability to save and load cloak colors

## Credits

This project was inspired by the magical world of Harry Potter. It demonstrates the practical application of computer vision techniques including:

- Real-time image processing
- Color space conversion (RGB to HSV)
- Background subtraction
- Bitwise operations
- Morphological transformations

## License

This project is open source and available under the MIT License.
