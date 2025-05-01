# ♟️ Chess Piece Detection and Move Prediction

This project combines object detection and chess AI to analyze a chessboard image and suggest the next best moves. It uses a YOLOv8 model to detect chess pieces from an uploaded image and the Stockfish engine to generate strategic move suggestions. A Gradio interface allows users to interactively visualize the current board and the recommended moves.

## 📦 Features

- Detects and classifies chess pieces (e.g., white-rook, black-pawn) from a board image.
- Automatically maps detected pieces to FEN-compliant board squares.
- Uses the Stockfish engine to analyze the board and suggest the best `N` moves.
- Displays detected pieces and planned moves with arrows and labels.
- Web-based interface via Gradio for easy user interaction.

## 🧰 Requirements

- Python 3.7 or later
- Installed Stockfish chess engine (available at [stockfishchess.org](https://stockfishchess.org/download/))
- Python libraries (install with the command below):

```bash
pip install gradio ultralytics opencv-python matplotlib chess
```

🚀 How to Run

1)Clone or download this repository.

2)Ensure the YOLO model (chess.pt) is located at the correct path (e.g., /content/chess.pt).

3)Run the application:

```bash
python app.py
```
4)Gradio will launch a browser window or link. Upload a chessboard image and select the number of future moves you want to visualize.

🖼 Interface Overview

-Image Upload: Upload a top-down photo or screenshot of a chessboard.

-Move Count Slider: Select how many upcoming moves to generate (1–5).

-Visualization: The board will be shown with piece annotations and colored arrows indicating the best next moves.

⚙️ How It Works

The YOLO model detects all pieces and returns bounding boxes, labels, and confidence scores.

The chessboard is divided into an 8×8 grid, and detected pieces are assigned to the closest square.

A virtual board is constructed using the python-chess library.

The Stockfish engine analyzes the current position and suggests moves.

Detected pieces and suggested moves are visualized on the image using Matplotlib.


📌 Example

Upload a board image (e.g., from a physical board or a digital screenshot).

The interface highlights each detected piece with its label.

Arrows show the best moves computed by Stockfish in order.

📂 Notes

Accurate detection depends on the quality and angle of the input image.

Make sure the chessboard is clearly visible and not obscured.

Custom YOLO weights (chess.pt) must be trained or downloaded separately.

You can update the color and annotation style in the process_image function if needed.

📝 License

This project is open-source and available under the MIT License.


🙏 Acknowledgements
YOLOv8 by Ultralytics

Stockfish chess engine

Gradio for rapid prototyping

Python-Chess library
