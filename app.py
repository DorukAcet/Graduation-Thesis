import gradio as gr
from ultralytics import YOLO
from mmdet.apis import DetInferencer
from PIL import Image
import cv2
import numpy as np
import os
import tempfile

# FRCNN için
frcnn_class_names = [
    'Bream', 'Mackerel', 'Red-Mullet', 'Red-Sea-Bream',
    'Sea-Bass', 'Sprat', 'Striped-Red-Mullet', 'Trout', 'Shrimp'
]

# YOLO için wiki links (lowercase ve alt çizgiyle)
wiki_links = {
    "bream": "https://en.wikipedia.org/wiki/Bream",
    "mackerel": "https://en.wikipedia.org/wiki/Mackerel",
    "red_mullet": "https://en.wikipedia.org/wiki/Red_mullet",
    "red_sea_bream": "https://en.wikipedia.org/wiki/Red_sea_bream",
    "sea_bass": "https://en.wikipedia.org/wiki/Sea_bass",
    "sprat": "https://en.wikipedia.org/wiki/Sprat",
    "striped_red_mullet": "https://en.wikipedia.org/wiki/Striped_red_mullet",
    "trout": "https://en.wikipedia.org/wiki/Trout",
    "shrimp": "https://en.wikipedia.org/wiki/Shrimp"
}

# YOLO yükle
try:
    yolo_model = YOLO("yolov11_best.pt")
    yolo_class_names = list(yolo_model.names.values())  # Mapping garanti!
except Exception as e:
    print(f"❌ YOLOv11 model yüklenemedi: {e}")
    yolo_model = None
    yolo_class_names = []

# FRCNN yükle
try:
    frcnn_infer = DetInferencer(
        model='faster_rcnn_custom.py',
        weights='epoch_15.pth',
        device='cpu'
    )
    print("✅ Faster R-CNN checkpoint başarıyla yüklendi!")
except Exception as e:
    print(f"❌ Faster R-CNN yüklenemedi: {e}")
    frcnn_infer = None

def detect_yolov11(image, conf_threshold):
    if yolo_model is None:
        return [], []
    results = yolo_model(image, conf=conf_threshold)
    boxes = results[0].boxes.data.cpu().numpy() if len(results) > 0 else []
    yolo_boxes = []
    yolo_classes = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        if conf < conf_threshold or int(cls) >= len(yolo_class_names):
            continue
        cls_name = yolo_class_names[int(cls)]
        yolo_classes.append(cls_name)
        yolo_boxes.append((int(x1), int(y1), int(x2), int(y2), cls_name, conf))
    return yolo_boxes, yolo_classes

def detect_fasterrcnn(image, conf_threshold):
    if frcnn_infer is None:
        return [], []
    img_array = np.array(image)[:, :, ::-1]
    results = frcnn_infer(img_array, show=False, out_dir=None, no_save_vis=True, pred_score_thr=conf_threshold)
    preds = results.get('predictions', [])
    frcnn_boxes = []
    frcnn_classes = []
    if preds and len(preds[0]['labels']) > 0:
        zipped = list(zip(preds[0]['labels'], preds[0]['scores'], preds[0]['bboxes']))
        if len(zipped) > 0:
            max_box = max(zipped, key=lambda x: x[1])
            label, score, box = max_box
            if score >= conf_threshold and int(label) < len(frcnn_class_names):
                cls_name = frcnn_class_names[int(label)]
                frcnn_classes.append(cls_name)
                x1, y1, x2, y2 = map(int, box)
                frcnn_boxes.append((x1, y1, x2, y2, cls_name, score))
    return frcnn_boxes, frcnn_classes

def compare_models(image, conf_threshold):
    yolo_boxes, yolo_classes = detect_yolov11(image, conf_threshold)
    frcnn_boxes, frcnn_classes = detect_fasterrcnn(image, conf_threshold)
    yolo_classes = list(set(yolo_classes))
    frcnn_classes = list(set(frcnn_classes))
    combined_img = np.array(image).copy()
    for x1, y1, x2, y2, cls_name, conf in yolo_boxes:
        cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(combined_img, f"YOLO: {cls_name} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for x1, y1, x2, y2, cls_name, score in frcnn_boxes:
        cv2.rectangle(combined_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(combined_img, f"FRCNN: {cls_name} {score:.2f}", (x1, y1-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    combined_img_pil = Image.fromarray(combined_img)
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "output.jpg")
    combined_img_pil.save(output_path)
    return combined_img_pil, yolo_classes, frcnn_classes, output_path

def create_links(class_list):
    # Her bir class'ı küçük harfe ve alt çizgiye çevir (YOLO mapping ile tam uyumlu)
    return [
        f'<a href="{wiki_links.get(c.lower(), "#")}" target="_blank">{c}</a>' if c.lower() in wiki_links else c
        for c in class_list
    ]

with gr.Blocks() as demo:
    gr.Markdown("# 🐟 Fish Detection with YOLOv11 & Faster R-CNN")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image", type="pil")
            threshold = gr.Slider(0, 1, value=0.25, step=0.01, label="Confidence Threshold")
            btn = gr.Button("Detect")
        with gr.Column():
            output_img = gr.Image(label="YOLOv11 vs Faster R-CNN (Overlayed)")
            gr.Markdown("### Detected Classes (Click for Wikipedia)")
            yolo_list = gr.HTML(label="YOLOv11 Detected")
            frcnn_list = gr.HTML(label="Faster R-CNN Detected")
            download_btn = gr.File(label="Download Combined Output")

    def run_all(image, threshold):
        combined_img, yolo_classes, frcnn_classes, path = compare_models(image, threshold)
        # Link fonksiyonunda küçük harfe ve alt çizgiye çevirerek %100 uyum sağlıyoruz
        return combined_img, "<br>".join(yolo_classes), "<br>".join(create_links(frcnn_classes)), path


    btn.click(
        run_all,
        inputs=[image_input, threshold],
        outputs=[output_img, yolo_list, frcnn_list, download_btn]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
