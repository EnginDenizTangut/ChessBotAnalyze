# ♟️ Chess Piece Detection and Move Prediction

This project combines computer vision with a chess engine to detect pieces from an image of a chessboard and predict the next best moves. It uses a custom-trained **YOLO** model for object detection, the **Stockfish** engine for move analysis, and **Gradio** for an interactive web UI.

---

## 🚀 Features

- Detect chess pieces (with color and type) from a real or digital chessboard image.
- Map detected pieces to their correct board squares.
- Predict the top N best moves using the Stockfish engine.
- Visualize detected pieces and predicted moves interactively.
- User-friendly Gradio interface.

---

## 🛠️ Requirements

- Python 3.7+
- [Stockfish Chess Engine](https://stockfishchess.org/download/) installed locally

### Python Dependencies:

Install with:

```bash
pip install gradio ultralytics opencv-python matplotlib chess
