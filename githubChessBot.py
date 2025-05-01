import gradio as gr
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import chess
import chess.engine

# Modeli yükle
model = YOLO('/content/chess.pt')

# Görsel işleme fonksiyonu
def process_image(image_path, num_moves):
    # Resmi yükle ve modele ver
    results = model(image_path, conf=0.7, iou=0.5)

    # Resmi oku ve boyutları ayarla
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_size = (800, 800)  # 800x800 piksel
    image = cv2.resize(image, target_size)
    image_height, image_width, _ = image.shape

    # Kare boyutları
    cell_size_x = image_width / 8
    cell_size_y = image_height / 8

    # Algılanan taşlar
    boxes = results[0].boxes.xywh.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    labels = [results[0].names[class_id] for class_id in class_ids]

    # Her kare için taşları eşle
    square_to_piece = defaultdict(list)
    for label, box, confidence in zip(labels, boxes, confidences):
        x_center, y_center, _, _ = box
        col = int(x_center // cell_size_x)
        row = int(y_center // cell_size_y)
        if 0 <= col < 8 and 0 <= row < 8:
            square = f"{chr(col + ord('a'))}{8 - row}"
            square_to_piece[square].append((label, confidence))

    # En iyi taşları seç
    final_pieces = {}
    for square, pieces in square_to_piece.items():
        if pieces:
            best_piece = max(pieces, key=lambda x: x[1])
            final_pieces[square] = best_piece[0]

    # Satranç tahtasını oluştur
    board = chess.Board(None)
    piece_map = {
        'pawn': chess.PAWN,
        'knight': chess.KNIGHT,
        'bishop': chess.BISHOP,
        'rook': chess.ROOK,
        'queen': chess.QUEEN,
        'king': chess.KING
    }

    for square, piece in final_pieces.items():
        color, piece_type = piece.split('-')
        chess_square = chess.parse_square(square)
        board.set_piece_at(chess_square, chess.Piece(piece_map[piece_type], chess.WHITE if color == 'white' else chess.BLACK))

    # Stockfish motorunu aç
    engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

    # Plan hamlelerini bul
    plan_moves = []
    current_board = board.copy()

    for _ in range(num_moves):
        info = engine.analyse(current_board, chess.engine.Limit(time=1.0))
        if 'pv' in info:
            move = info['pv'][0]
            plan_moves.append(move)
            current_board.push(move)
        else:
            break

    engine.quit()

    # Görselleştirme
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    # Kareleri çiz
    for row in range(8):
        for col in range(8):
            x = col * cell_size_x
            y = row * cell_size_y
            rect = patches.Rectangle((x, y), cell_size_x, cell_size_y,
                                    linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    # Taşları işaretle
    for square, piece in final_pieces.items():
        col = ord(square[0]) - ord('a')
        row = 8 - int(square[1])
        x_center = (col + 0.5) * cell_size_x
        y_center = (row + 0.5) * cell_size_y
        ax.text(x_center, y_center, piece.split('-')[-1][0].upper(),
                ha='center', va='center', fontsize=12, color='blue',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Plan hamlelerini çiz
    colors = ['green', 'blue', 'purple', 'orange', 'cyan', 'red', 'yellow', 'pink', 'brown', 'gray']
    for idx, move in enumerate(plan_moves):
        from_sq = move.from_square
        to_sq = move.to_square

        from_col = chess.square_file(from_sq)
        from_row = 7 - chess.square_rank(from_sq)
        to_col = chess.square_file(to_sq)
        to_row = 7 - chess.square_rank(to_sq)

        color = colors[idx % len(colors)]

        ax.annotate("",
                    xy=((to_col + 0.5) * cell_size_x, (to_row + 0.5) * cell_size_y),
                    xytext=((from_col + 0.5) * cell_size_x, (from_row + 0.5) * cell_size_y),
                    arrowprops=dict(arrowstyle="->", color=color, linewidth=3))

        ax.text((from_col + to_col + 1) * cell_size_x / 2,
                (from_row + to_row + 1) * cell_size_y / 2,
                f"{idx + 1}",
                ha='center', va='center',
                bbox=dict(facecolor=color, alpha=0.5))

    ax.axis('off')
    plt.title(f"Planlanan İlk {num_moves} Hamle")
    plt.close(fig)

    return fig

# Gradio Arayüzü
with gr.Blocks() as demo:
    gr.Markdown("### 🧩 Satranç Tahtası Görselleştirme ve Plan Hamleleri")

    with gr.Row():
        image_input = gr.Image(label="Satranç Görseli", type="filepath")
        num_moves_input = gr.Slider(minimum=1, maximum=5, value=3, label="Planlanan Hamle Sayısı")

    output = gr.Plot()

    submit_btn = gr.Button("Görselleştir")

    submit_btn.click(fn=process_image, inputs=[image_input, num_moves_input], outputs=output)

demo.launch(share=True, debug=True)
