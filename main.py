import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score
import io
import os

# Face Detection Functions
def display_images(images, captions, n_cols=4):
    n_rows = (len(images) - 1) // n_cols + 1
    rows = [st.columns(n_cols) for _ in range(n_rows)]
    for i, (image, caption) in enumerate(zip(images, captions)):
        with rows[i // n_cols][i % n_cols]:
            st.image(image, caption=caption, use_column_width=True)

def detect_faces(image, cascade_path):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    img_with_faces = np.array(image.copy())
    for (x, y, w, h) in faces:
        cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return Image.fromarray(img_with_faces), len(faces)

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def calculate_dice(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    dice = (2 * inter_area) / (box1_area + box2_area) if (box1_area + box2_area) > 0 else 0
    return dice

def evaluate_detection(image_path, our_cascade_path, opencv_cascade_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    our_cascade = cv2.CascadeClassifier(our_cascade_path)
    opencv_cascade = cv2.CascadeClassifier(opencv_cascade_path)
    
    our_faces = our_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    opencv_faces = opencv_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    iou_scores = []
    dice_scores = []
    
    for our_face in our_faces:
        best_iou = 0
        best_dice = 0
        for opencv_face in opencv_faces:
            iou = calculate_iou(our_face, opencv_face)
            dice = calculate_dice(our_face, opencv_face)
            best_iou = max(best_iou, iou)
            best_dice = max(best_dice, dice)
        iou_scores.append(best_iou)
        dice_scores.append(best_dice)
    
    avg_iou = np.mean(iou_scores) if iou_scores else 0
    avg_dice = np.mean(dice_scores) if dice_scores else 0
    
    return avg_iou, avg_dice, image, our_faces, opencv_faces

def evaluate_multiple_images(image_folder, our_cascade_path, opencv_cascade_path):
    iou_scores = []
    dice_scores = []
    image_names = []

    for image_name in os.listdir(image_folder):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_name)
            avg_iou, avg_dice, _, _, _ = evaluate_detection(image_path, our_cascade_path, opencv_cascade_path)
            iou_scores.append(avg_iou)
            dice_scores.append(avg_dice)
            image_names.append(image_name)

    return iou_scores, dice_scores, image_names

def face_detection_app():
    st.title("Face Detection using Haar Features")

    # Phần 1: Dataset
    st.header("1. Dataset")
    st.subheader("Positive Images")
    pos_images = [Image.open(f"images/face/{img}") for img in os.listdir("images/face")[:4]]
    pos_captions = [f"Positive {i+1}" for i in range(len(pos_images))]
    display_images(pos_images, pos_captions)

    st.subheader("Negative Images")
    neg_images = [Image.open(f"images/non_face/{img}") for img in os.listdir("images/non_face")[:4]]
    neg_captions = [f"Negative {i+1}" for i in range(len(neg_images))]
    display_images(neg_images, neg_captions)

    # Phần 2: Giới thiệu về phương pháp đánh giá
    st.header("2. Evaluation Metrics")
    st.subheader("IoU (Intersection over Union)")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("""
        IoU là một metric đánh giá độ chính xác của object detection bằng cách so sánh vùng overlap 
        giữa bounding box dự đoán và ground truth. IoU được tính bằng công thức:

        IoU = Area of Overlap / Area of Union

        Giá trị IoU nằm trong khoảng [0, 1], với 1 là dự đoán hoàn hảo.
        """)
    with col2:
        iou_image = Image.open("images/IoU.jpg")
        st.image(iou_image, caption="IoU Illustration", use_column_width=True)

    st.subheader("Dice Coefficient")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("""
        Dice Coefficient là một metric tương tự IoU, nhưng nhấn mạnh vào vùng overlap. Công thức:

        Dice = 2 * Area of Overlap / (Area of Prediction + Area of Ground Truth)

        Giá trị Dice cũng nằm trong khoảng [0, 1], với 1 là dự đoán hoàn hảo.
        """)
    with col2:
        dice_image = Image.open("images/Dice.jpg")
        st.image(dice_image, caption="Dice Coefficient Illustration", use_column_width=True)

    # Phần 3: Kết quả đánh giá
    st.header("3. Evaluation Results")
    try:
        results = np.load("evaluation_results.npz")
        iou_scores = results['iou_scores']
        dice_scores = results['dice_scores']
        image_names = results['image_names']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.bar(range(len(iou_scores)), iou_scores)
        ax1.set_title("IoU Scores")
        ax1.set_xlabel("Image")
        ax1.set_ylabel("IoU")
        ax1.set_xticks(range(len(iou_scores)))
        ax1.set_xticklabels(image_names, rotation=90)

        ax2.bar(range(len(dice_scores)), dice_scores)
        ax2.set_title("Dice Coefficient Scores")
        ax2.set_xlabel("Image")
        ax2.set_ylabel("Dice Coefficient")
        ax2.set_xticks(range(len(dice_scores)))
        ax2.set_xticklabels(image_names, rotation=90)

        plt.tight_layout()
        st.pyplot(fig)

        avg_iou = results['iou_scores'].mean()
        avg_dice = results['dice_scores'].mean()
        st.write(f"Average IoU: {avg_iou:.4f}")
        st.write(f"Average Dice Coefficient: {avg_dice:.4f}")
    except FileNotFoundError:
        st.write("Evaluation results file not found. Please run the evaluation script first.")

    # Phần 5: Nhận diện gương mặt
    st.header("5. Face Detection")
    uploaded_file = st.file_uploader("Choose an image for face detection...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Faces"):
            result_img, face_count = detect_faces(image, "data/cascade.xml")
            st.image(result_img, caption=f"Detected {face_count} faces", use_column_width=True)

# GrabCut Functions
def draw_rectangle(image, rect):
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    left, top, width, height = rect
    draw.rectangle([left, top, left + width, top + height], outline="blue", width=2)
    return np.array(img_pil)

def grabcut_segmentation(image, rect):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return image * mask2[:, :, np.newaxis]

def image_to_bytes(image):
    img = Image.fromarray(image)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

def grabcut_app():
    st.header("GrabCut Segmentation")
    
    st.markdown("""
        <style>
        body {
            background-image: url('38268813.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        h1 {
            color: #ff5733;
            text-align: center;
            font-size: 48px;
            font-family: 'Arial', sans-serif;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
        }
        .animated-text {
            animation: bounce 2s infinite;
            color: red;
            text-align: center;
            font-size: 36px;
        }
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% {
                transform: translateY(0);
            }
            40% {
                transform: translateY(-30px);
            }
            60% {
                transform: translateY(-15px);
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.write("Chọn cách nhập tọa độ vùng hình chữ nhật:")
        
        input_mode = st.radio("Chọn phương thức nhập:", ("Thanh kéo (Slider)", "Nhập thủ công (Number Input)"))

        if input_mode == "Thanh kéo (Slider)":
            left = st.slider("Tọa độ X của góc trái trên", 0, image_np.shape[1], 0)
            top = st.slider("Tọa độ Y của góc trái trên", 0, image_np.shape[0], 0)
            right = st.slider("Tọa độ X của góc phải dưới", 0, image_np.shape[1], image_np.shape[1])
            bottom = st.slider("Tọa độ Y của góc phải dưới", 0, image_np.shape[0], image_np.shape[0])
        else:
            left = st.number_input("Nhập X của góc trái trên", 0, image_np.shape[1], 0)
            top = st.number_input("Nhập Y của góc trái trên", 0, image_np.shape[0], 0)
            right = st.number_input("Nhập X của góc phải dưới", 0, image_np.shape[1], image_np.shape[1])
            bottom = st.number_input("Nhập Y của góc phải dưới", 0, image_np.shape[0], image_np.shape[0])

        rect = (left, top, right - left, bottom - top)
        
        img_with_rect = draw_rectangle(image_np, rect)
        st.image(img_with_rect, caption="Ảnh với hình chữ nhật", use_column_width=True)

        if st.button("Áp dụng thuật toán GrabCut"):
            if rect:
                with st.spinner("Đang phân đoạn ảnh, vui lòng đợi..."):
                    segmented_image = grabcut_segmentation(image_np, rect)

                st.markdown("""
                    <h2 class="animated-text">✨ Kết quả ✨</h2>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.image(img_with_rect, caption="Ảnh gốc với hình chữ nhật", use_column_width=True)

                with col2:
                    st.image(segmented_image, caption="Ảnh phân đoạn với GrabCut", use_column_width=True)
                    
                segmented_image_bytes = image_to_bytes(segmented_image)
                st.download_button(
                    label="Tải về ảnh phân đoạn",
                    data=segmented_image_bytes,
                    file_name="segmented_image.png",
                    mime="image/png"
                )
            else:
                st.write("Vui lòng nhập tọa độ hình chữ nhật.")

# Watershed Functions
def process_image(img, threshold_factor):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, threshold_factor*dist_transform.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    return unknown

def calculate_metrics(prediction, ground_truth):
    prediction = (prediction > 0).astype(int)
    ground_truth = (ground_truth > 0).astype(int)
    
    prediction = prediction.flatten()
    ground_truth = ground_truth.flatten()
    
    iou = jaccard_score(ground_truth, prediction, average='binary')
    dice = f1_score(ground_truth, prediction, average='binary')
    
    return iou, dice

def watershed_app():
    st.title("Watershed Algorithm Analysis")

    # Phần 1: Dataset
    st.header("1. Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Images")
        st.image("images/WS/training/ndata616.jpg", caption="Training Image 1")
        st.image("images/WS/training/ndata617.jpg", caption="Training Image 2")
    with col2:
        st.subheader("Test Images")
        st.image("images/WS/gt/ndata616gt.jpg", caption="Test Image 1")
        st.image("images/WS/gt/ndata617gt.png", caption="Test Image 2")

    # Phần 2: Metrics
    st.header("2. Evaluation Metrics")
    st.subheader("IoU (Intersection over Union)")
    st.write("IoU measures the overlap between the predicted segmentation and the ground truth.")
    st.image("images/IoU.png", caption="IoU Visualization")

    st.subheader("Dice Coefficient")
    st.write("Dice coefficient is a statistical measure of the overlap between two sets.")
    st.image("images/Dice.jpg", caption="Dice Coefficient Visualization")

    # Phần 3: Parameters
    st.header("3. Algorithm Parameters")
    st.write("Threshold range: 0.1 to 1.0")
    st.write("Threshold step: 0.1")
    st.write("Distance Transform: cv2.DIST_L2")
    st.write("Kernel size for morphological operations: 3x3")

    # Phần 4: Watershed Application
    st.header("4. Apply Watershed Algorithm")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Apply Watershed"):
            thresholds = np.arange(0.1, 1.1, 0.1)
            best_iou = 0
            best_threshold = 0
            best_result = None
            
            for threshold in thresholds:
                result = process_image(img_array, threshold)
                iou, _ = calculate_metrics(result, img_array[:,:,0])  # Assuming grayscale ground truth
                
                if iou > best_iou:
                    best_iou = iou
                    best_threshold = threshold
                    best_result = result
            
            st.write(f"Best Threshold: {best_threshold:.1f}")
            st.write(f"Best IoU: {best_iou:.4f}")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(img_array)
            ax1.set_title("Original Image")
            ax2.imshow(best_result, cmap='gray')
            ax2.set_title("Watershed Result")
            st.pyplot(fig)

    st.sidebar.header("About")
    st.sidebar.info("This app demonstrates the Watershed algorithm for image segmentation and evaluates its performance using IoU and Dice coefficient metrics.")

def main():
    st.sidebar.title("OpenCV Applications")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Face Detection", "GrabCut Segmentation", "Watershed Segmentation"])
    
    if app_mode == "Face Detection":
        face_detection_app()
    elif app_mode == "GrabCut Segmentation":
        grabcut_app()
    elif app_mode == "Watershed Segmentation":
        watershed_app()

if __name__ == "__main__":
    main()
