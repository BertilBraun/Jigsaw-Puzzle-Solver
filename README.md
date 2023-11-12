# Jigsaw Puzzle Solver

## Overview

The Jigsaw Puzzle Solver is a tool for automatically determining the correct placement of jigsaw puzzle pieces using image processing and computer vision. This tool is implemented in Python, utilizing the OpenCV library and the CLIP model for image recognition. There are two main components: a command-line solver and a web interface provided via Flask.

## Features

- Command-line puzzle solving with manual input of puzzle piece paths.
- Flask web application for uploading images and receiving immediate placement results.
- Utilization of OpenCV for image manipulation and the CLIP model for advanced image recognition.

## Components

### Base Solver (`solver.py`)

The base solver is a Python script that can be used via the command line by hardcoding the puzzle image path and dimensions.

#### Usage

To solve a puzzle, run:
```python
puzzle_solver("path_to_puzzle_image.png", (piece_width, piece_height))
```
The solver will prompt for the paths to individual piece images and output the matching locations.

### Web Interface (`web_solver.py`)

The web interface allows users to upload a puzzle image through a web form and receive the matching location within the interface.

#### Usage

1. Start the Flask app:
   ```sh
   python web_solver.py
   ```
2. Open a web browser and navigate to `localhost:5000`.
3. Follow the on-screen instructions to upload a puzzle piece image and receive the location in the puzzle.

## Installation

### Prerequisites

- Python 3.x
- Flask
- OpenCV
- CLIP model

### Setup

1. Clone the repository: `git clone https://github.com/BertilBraun/Jigsaw-Puzzle-Solver.git`
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the base solver or web interface as needed.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

## Contact

For queries or support, open an issue in the repository or reach out to the maintainers.
