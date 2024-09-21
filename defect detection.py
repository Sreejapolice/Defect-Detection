import cv2
import numpy as np

def preprocess_image(image_path):
    """Load and preprocess the image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (500, 500))
    return image, resized_image

def detect_changes(reference_image_path, captured_image_path, threshold=50):
    """Detect changes between the reference and captured images."""
    reference_image, resized_reference = preprocess_image(reference_image_path)
    captured_image, resized_captured = preprocess_image(captured_image_path)
   
    if resized_reference is None or resized_captured is None:
        return

    difference = cv2.absdiff(resized_reference, resized_captured)
    _, thresh_diff = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
   
    contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_image = cv2.cvtColor(resized_captured, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(annotated_image, contours, -1, (0, 255, 0), 2)
   
    cv2.imshow('Difference', difference)
    cv2.imshow('Thresholded Difference', thresh_diff)
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(5000)  # Wait for 5 seconds before closing the windows
    cv2.destroyAllWindows()
   
    num_changes = len(contours)
    print(f'Number of changes detected: {num_changes}')
    if num_changes > 0:
        print('Changes detected in the image. Please review the annotated image for details.')

def detect_color_changes(reference_image_path, captured_image_path):
    """Detect color changes between the reference and captured images."""
    ref_image = cv2.imread(reference_image_path)
    cap_image = cv2.imread(captured_image_path)
   
    if ref_image is None or cap_image is None:
        print("Error: Unable to load one of the images.")
        return

    # Resize images to the same dimensions
    ref_image_resized = cv2.resize(ref_image, (500, 500))
    cap_image_resized = cv2.resize(cap_image, (500, 500))
   
    ref_hsv = cv2.cvtColor(ref_image_resized, cv2.COLOR_BGR2HSV)
    cap_hsv = cv2.cvtColor(cap_image_resized, cv2.COLOR_BGR2HSV)
   
    h_diff = cv2.absdiff(ref_hsv[:, :, 0], cap_hsv[:, :, 0])
    s_diff = cv2.absdiff(ref_hsv[:, :, 1], cap_hsv[:, :, 1])
    v_diff = cv2.absdiff(ref_hsv[:, :, 2], cap_hsv[:, :, 2])
   
    diff_image = cv2.merge([h_diff, s_diff, v_diff])
    cv2.imshow('Color Difference', diff_image)
    cv2.waitKey(5000)  # Wait for 5 seconds before closing the windows
    cv2.destroyAllWindows()
   
    print('Color changes detected. Review the color difference image for details.')

def detect_angle(reference_image_path, captured_image_path):
    """Detect angle changes between the reference and captured images."""
    ref_image = cv2.imread(reference_image_path)
    cap_image = cv2.imread(captured_image_path)

    if ref_image is None or cap_image is None:
        print("Error: Unable to load one of the images.")
        return

    # Convert images to grayscale and find corners
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    cap_gray = cv2.cvtColor(cap_image, cv2.COLOR_BGR2GRAY)
   
    ref_corners = cv2.goodFeaturesToTrack(ref_gray, 4, 0.01, 10)
    cap_corners = cv2.goodFeaturesToTrack(cap_gray, 4, 0.01, 10)

    if ref_corners is None or cap_corners is None:
        print("Not enough feature points detected.")
        return
   
    # Convert corners to integer coordinates
    ref_corners = np.int32(ref_corners)
    cap_corners = np.int32(cap_corners)
   
    if len(ref_corners) == 4 and len(cap_corners) == 4:
        ref_points = np.array([p[0] for p in ref_corners], dtype=np.float32)
        cap_points = np.array([p[0] for p in cap_corners], dtype=np.float32)
       
        h, _ = cv2.findHomography(ref_points, cap_points)
        angle = np.arctan2(h[1, 0], h[0, 0]) * 180.0 / np.pi
        print(f'Detected angle change: {angle:.2f} degrees')
        if abs(angle) > 1.0:  # Example threshold for angle change
            print(f'Angle change detected: {angle:.2f} degrees. Please review the angle change.')
        else:
            print('No significant angle change detected.')
    else:
        print('Insufficient feature points to detect angle change.')

def detect_rust(image_path):
    """Detect rust or corrosion in the image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
   
    lower_rust = np.array([10, 100, 100])
    upper_rust = np.array([30, 255, 255])
   
    rust_mask = cv2.inRange(hsv_image, lower_rust, upper_rust)
    rust_detection = cv2.bitwise_and(image, image, mask=rust_mask)
   
    cv2.imshow('Rust Detection', rust_detection)
    cv2.waitKey(5000)  # Wait for 5 seconds before closing the windows
    cv2.destroyAllWindows()
   
    rust_pixels = cv2.countNonZero(rust_mask)
    if rust_pixels > 0:
        print(f'Rust detected. Review the rust detection image for details.')
    else:
        print('No rust detected.')

def detect_objects(image_path):
    """Detect objects using YOLO."""
    weights_path = "C:/Users/bhara/OneDrive/Desktop/yolo/yolov3.weights"
    config_path = "C:/Users/bhara/OneDrive/Desktop/yolo/yolov3.cfg"
    names_path = "C:/Users/bhara/OneDrive/Desktop/yolo/coco.names"

    # Load YOLO
    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    out_layer_indices = net.getUnconnectedOutLayers()

    if len(out_layer_indices.shape) == 1:
        out_layer_indices = out_layer_indices.flatten()
    output_layers = [layer_names[i - 1] for i in out_layer_indices]

    try:
        with open(names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: {names_path} file not found.")
        return

    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    warning_issued = False  # Flag to ensure only one warning is issued

    for out in outs:
        for detection in out:
            for obj in detection:
                obj = np.array(obj)  # Ensure obj is an array
                if obj.ndim == 1 and obj.size >= 5:  # Ensure obj has at least 5 elements
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(obj[0] * width)
                        center_y = int(obj[1] * height)
                        w = int(obj[2] * width)
                        h = int(obj[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                else:
                    if not warning_issued:
                        print(f"Warning: Detected object does not have enough elements. Size: {obj.size}, Elements: {obj}")
                        warning_issued = True  # Set flag to true after issuing a warning

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if indices is not None and len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection", image)
    cv2.waitKey(5000)  # Wait for 5 seconds before closing the windows
    cv2.destroyAllWindows()

    num_objects = len(indices) if indices is not None else 0
    if num_objects > 0:
        print(f'Number of objects detected: {num_objects}')
        print('Objects detected. Review the object detection image for details.')
    else:
        print('No objects detected.')

# Paths to your images
reference_image_path = r'C:\Users\bhara\Downloads\WhatsApp Image 2024-09-12 at 12.44.40 AM.jpeg'
captured_image_path = r'C:\Users\bhara\Downloads\WhatsApp Image 2024-09-12 at 12.44.40 AM (1).jpeg'

# Perform all detections
detect_changes(reference_image_path, captured_image_path)
detect_color_changes(reference_image_path, captured_image_path)
detect_angle(reference_image_path, captured_image_path)
detect_rust(captured_image_path)
detect_objects(captured_image_path)
