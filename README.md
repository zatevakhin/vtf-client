# vtf-client

This project provides a client for the [vtf](https://github.com/zatevakhin/vtf) Godot project. It uses a webcam to capture facial expressions, head movements, and iris movements, and sends them to the Godot server in real-time.

## How It Works

The client uses `opencv-python` to capture video from a webcam, `mediapipe` to detect facial landmarks, and `deepface` to analyze emotions. The data is then sent to the Godot server using `websockets`.

The server-side implementation in the `vtf` project listens for WebSocket connections and applies the received data to a 3D character model.

### Data Sent to the Server

The client sends the following data to the server in a JSON format:

-   `id`: The user/character ID.
-   `iris`: The position of the irises.
-   `rotation`: The rotation of the head as a quaternion.
-   `blend`: A dictionary of blend shapes for facial expressions.
-   `emotion`: A dictionary of emotions and their values.
-   `dominant_emotion`: The dominant emotion.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/zatevakhin/vtf-client
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Make sure the `vtf` Godot project is running and the WebSocket server is active.
2.  Run the client:
    ```bash
    python main.py --user-id <your-character-id>
    ```

    Replace `<your-character-id>` with the ID of the character you want to control in the Godot scene.

    You can also specify the camera to use, the host, and the port:

    ```bash
    python main.py --camera 0 --host localhost --port 8082 --user-id <your-character-id>
    ```

    -   `--camera`: The index of the camera to use (default: `0`).
    -   `--host`: The host of the WebSocket server (default: `localhost`).
    -   `--port`: The port of the WebSocket server (default: `8082`).
    -   `--user-id`: The ID of the character to control.
