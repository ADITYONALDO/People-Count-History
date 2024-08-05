# People Detection and Face Recognition

This project is a real-time people detection and face recognition system using OpenCV and SQLite. The system uses the LBPH (Local Binary Patterns Histograms) cascade classifier for face detection and a custom method for face recognition. It stores detection events in a SQLite database, recording the name of the detected person and the timestamps of their visits.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Database Schema](#database-schema)
- [Notes](#notes)
- [License](#license)

## Requirements

- Python 3.x
- OpenCV
- NumPy
- SQLite3

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/people-detection.git
    cd people-detection
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Prepare the known faces directory:**
    - Create a directory named `known_faces` in the project root.
    - Add images of known people to this directory. The image filenames should reflect the names of the people (e.g., `name1.jpg`, `name2.jpg`, `alex1.png`, etc.).

## Usage

1. **Run the script:**
    ```sh
    python detection.py
    ```

2. **Face detection and recognition:**
    - The system will start capturing video from your webcam.
    - It will detect faces in the video stream and recognize known faces based on the images in the `known_faces` directory.
    - The detected faces and their names will be displayed in the video window.

3. **Database updates:**
    - The system will log the visits of recognized people in the SQLite database (`people_detection.db`).
    - For each person, it will store the first and last visit timestamps of the day.

## Code Explanation

- **Initialization:**
    - The database is initialized and a table named `visits` is created to store detection events.
    - The LBPH cascade classifier is loaded for face detection.

- **Loading known faces:**
    - The `load_known_faces` function reads images from the `known_faces` directory, detects faces, and stores their encodings and names.

- **Face detection and recognition:**
    - The `detect_and_recognize` function processes each video frame, detects faces, and compares them with known face encodings to recognize them.

- **Database operations:**
    - The `update_or_insert_visit` function updates the database with the visit information, ensuring only one entry per person per day and updating timestamps as needed.

## Database Schema

- **visits Table:**
    - `ID` (INTEGER PRIMARY KEY AUTOINCREMENT): Unique identifier for each entry.
    - `Name` (TEXT): Name of the detected person.
    - `LastVisitTimestamp` (TEXT): Timestamp of the first visit of the day.
    - `NewVisitTimestamp` (TEXT): Timestamp of the last visit of the day.

## Notes

- The system uses a custom method for face recognition based on image encodings and L2 distance.
- Face detection is performed using the LBPH cascade classifier.
- Make sure the `known_faces` directory contains images with filenames representing the names of the people.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
