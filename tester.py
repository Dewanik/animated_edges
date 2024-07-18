import cv2
import numpy as np

def canny_edge_detection(frame):
    # Convert the frame to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and smoothen edges
    blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5)
    
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 70, 135)
    
    return edges

def detect_person(frame, net):
    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    
    # Set the blob as input to the network
    net.setInput(blob)
    
    # Perform forward pass to get the detections
    detections = net.forward()
    
    return detections

def detect_face(frame, face_net):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            return (startX, startY, endX - startX, endY - startY)
    return None

def main():
    # Load the pre-trained model and set up the network
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
    
    # Load face detection model
    face_net = cv2.dnn.readNetFromCaffe('deploy_face.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
    
    # Open the default webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print('Image not captured')
            break
        
        # Detect persons in the frame
        detections = detect_person(frame, net)
        
        (h, w) = frame.shape[:2]
        
        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                
                # Check if the detected object is a person (class ID for person in COCO dataset is 15)
                if class_id == 15:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Ensure the bounding box is within the frame
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w, endX), min(h, endY)
                    
                    # Perform Canny edge detection on the person
                    person = frame[startY:endY, startX:endX]
                    edges = canny_edge_detection(person)
                    
                    # Create a mask for the person
                    mask = np.zeros_like(frame)
                    
                    # Fill face, hands, and clothes with specific colors
                    face_color = [255, 204, 153]  # Light skin color
                    hand_color = [255, 204, 153]  # Light skin color
                    clothes_color = [0, 0, 255]   # Red color for clothes
                    
                    # Detect face
                    face_box = detect_face(person, face_net)
                    if face_box is not None:
                        (fx, fy, fw, fh) = face_box
                        mask[startY+fy:startY+fy+fh, startX+fx:startX+fx+fw] = face_color
                    
                    
        
        # Display the original frame and the processed frame
        cv2.imshow("Original", frame)
        cv2.imshow("Processed Frame", edges)
        
        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
