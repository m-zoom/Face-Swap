#!/usr/bin/env python3
import cv2
import numpy as np
import mediapipe as mp
import time
import os
import sys
import threading

class ImprovedFaceSwapper:
    def __init__(self, reference_image_path):
        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Face detection for initial reference face extraction
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        
        # Check if reference image exists
        if not os.path.exists(reference_image_path):
            print(f"Error: Reference image not found at '{reference_image_path}'")
            print(f"Current directory: {os.getcwd()}")
            print(f"Looking for: {os.path.abspath(reference_image_path)}")
            sys.exit(1)
            
        # Load reference image (the face to swap with)
        self.reference_img = cv2.imread(reference_image_path)
        if self.reference_img is None:
            print(f"Error: Could not load reference image '{reference_image_path}'")
            sys.exit(1)
            
        # Extract only the face from the reference image
        print("Extracting face from reference image...")
        self.reference_face = self.extract_face(self.reference_img)
        if self.reference_face is None:
            print("Error: Could not detect face in reference image. Using entire image.")
            self.reference_face = self.reference_img.copy()
        else:
            print("Face extracted successfully from reference image.")
            
        # Get reference face landmarks
        ref_rgb = cv2.cvtColor(self.reference_face, cv2.COLOR_BGR2RGB)
        self.ref_height, self.ref_width = self.reference_face.shape[:2]
        ref_results = self.face_mesh.process(ref_rgb)
        
        if not ref_results.multi_face_landmarks:
            print("Error: No facial landmarks detected in reference face")
            sys.exit(1)
            
        # Store reference landmarks
        self.reference_landmarks = []
        for landmark in ref_results.multi_face_landmarks[0].landmark:
            x, y = int(landmark.x * self.ref_width), int(landmark.y * self.ref_height)
            self.reference_landmarks.append((x, y))
        
        self.reference_landmarks = np.array(self.reference_landmarks)
        
        # Define indices for face parts (for better blending)
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        
        self.LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
        
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Create triangulation for face
        self.FACE_TRIANGLES = self.create_face_triangulation()
        
        # Webcam monitoring variables
        self.webcam_active = False
        self.cap = None
        self.monitor_thread = None
        self.running = True
    
    def extract_face(self, img):
        """Extract only the face region from an image"""
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_img)
        
        if not results.detections:
            return None
            
        # Get the bounding box of the first face
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        
        h, w, _ = img.shape
        x, y = int(bboxC.xmin * w), int(bboxC.ymin * h)
        width, height = int(bboxC.width * w), int(bboxC.height * h)
        
        # Add some margin around the face (20%)
        margin_x = int(width * 0.2)
        margin_y = int(height * 0.2)
        
        # Ensure we stay within image boundaries
        x_start = max(0, x - margin_x)
        y_start = max(0, y - margin_y)
        x_end = min(w, x + width + margin_x)
        y_end = min(h, y + height + margin_y)
        
        # Extract face region
        face_img = img[y_start:y_end, x_start:x_end].copy()
        
        return face_img
    
    def create_face_triangulation(self):
        """Create triangulation for the face mesh"""
        # Create a list of triangles for face mesh based on landmark indices
        # This is a simplified triangulation that works well for face swapping
        triangles = []
        
        # Face oval triangulation
        for i in range(len(self.FACE_OVAL) - 2):
            triangles.append([self.FACE_OVAL[0], self.FACE_OVAL[i + 1], self.FACE_OVAL[i + 2]])
            
        # Left eye triangulation
        for i in range(len(self.LEFT_EYE) - 2):
            triangles.append([self.LEFT_EYE[0], self.LEFT_EYE[i + 1], self.LEFT_EYE[i + 2]])
            
        # Right eye triangulation
        for i in range(len(self.RIGHT_EYE) - 2):
            triangles.append([self.RIGHT_EYE[0], self.RIGHT_EYE[i + 1], self.RIGHT_EYE[i + 2]])
            
        # Lips triangulation
        for i in range(len(self.LIPS) - 2):
            triangles.append([self.LIPS[0], self.LIPS[i + 1], self.LIPS[i + 2]])
            
        # Add more triangles for better face coverage
        additional_triangles = [
            # Forehead
            [10, 67, 109], [67, 109, 151], [109, 151, 338], [151, 338, 337],
            # Nose
            [168, 6, 197], [6, 197, 195], [197, 195, 5], [195, 5, 4],
            # Cheeks
            [116, 123, 147], [123, 147, 187], [147, 187, 207], [187, 207, 206],
            [346, 352, 376], [352, 376, 411], [376, 411, 423], [411, 423, 437],
            # Chin
            [17, 0, 18], [0, 18, 200], [18, 200, 199], [200, 199, 175]
        ]
        
        triangles.extend(additional_triangles)
        
        return triangles
        
    def get_face_mask(self, landmarks, img_shape):
        """Create a mask for the face area based on landmarks"""
        mask = np.zeros(img_shape, dtype=np.uint8)
        
        # Create face boundary from landmarks
        face_boundary = np.array([landmarks[i] for i in self.FACE_OVAL], dtype=np.int32)
        
        # Fill face region
        cv2.fillPoly(mask, [face_boundary], (255, 255, 255))
        
        return mask
    
    def apply_affine_transform(self, src, src_tri, dst_tri, size):
        """Apply affine transform to source image"""
        # Get the affine transform
        warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
        
        # Apply the affine transform
        dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), 
                            None, flags=cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_REFLECT_101)
        return dst
    
    def warp_triangle(self, img1, img2, t1, t2):
        """Warp triangular regions from img1 to img2"""
        # Find bounding rectangles for both triangles
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        
        # Offset the points for cropped image
        t1_rect = []
        t2_rect = []
        t2_rect_int = []
        
        for i in range(0, 3):
            t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
            t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        
        # Get cropped images
        img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        
        # Create mask
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)
        
        # Apply affine transform
        if img1_rect.size > 0:  # Check if the cropped image is not empty
            img2_rect = self.apply_affine_transform(img1_rect, t1_rect, t2_rect, (r2[2], r2[3]))
            
            # Multiply the warped image with the mask
            img2_rect = img2_rect * mask
            
            # Copy triangular region to the destination image
            img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1 - mask) + img2_rect
    
    def swap_faces(self, source_img, source_landmarks):
        """Swap faces between source and reference images"""
        # Create a copy of the source image
        img_result = source_img.copy()
        h, w = source_img.shape[:2]
        
        # Swap each triangle
        for triangle in self.FACE_TRIANGLES:
            # Check if indices are valid
            if max(triangle) >= len(source_landmarks) or max(triangle) >= len(self.reference_landmarks):
                continue
                
            # Get points for source triangle
            tri_src = np.array([
                source_landmarks[triangle[0]],
                source_landmarks[triangle[1]],
                source_landmarks[triangle[2]]
            ])
            
            # Get points for reference triangle
            tri_ref = np.array([
                self.reference_landmarks[triangle[0]],
                self.reference_landmarks[triangle[1]],
                self.reference_landmarks[triangle[2]]
            ])
            
            # Warp the triangle
            self.warp_triangle(self.reference_face, img_result, tri_ref, tri_src)
        
        # Create masks for seamless blending
        face_mask = self.get_face_mask(source_landmarks, source_img.shape)
        face_mask_gray = cv2.cvtColor(face_mask, cv2.COLOR_BGR2GRAY)
        
        # Get face center for seamless clone
        face_center = np.mean(np.array([source_landmarks[i] for i in self.FACE_OVAL]), axis=0).astype(int)
        
        # Apply seamless blending
        try:
            output = cv2.seamlessClone(img_result, source_img, face_mask_gray, tuple(face_center), cv2.NORMAL_CLONE)
        except cv2.error as e:
            print(f"Seamless clone error: {e}")
            # Fallback - use alpha blending instead
            alpha = 0.75  # Make the reference face more prominent
            beta = 1.0 - alpha
            
            # Apply alpha blending only where the mask is active
            mask_3ch = cv2.cvtColor(face_mask_gray, cv2.COLOR_GRAY2BGR) / 255.0
            output = cv2.addWeighted(img_result, alpha, source_img, beta, 0) * mask_3ch + source_img * (1 - mask_3ch)
            output = output.astype(np.uint8)
        
        # Enhanced blending for eyes and mouth to ensure they're very visible
        # Create special masks for eyes and mouth
        eyes_mouth_mask = np.zeros_like(source_img)
        
        # Draw eyes and mouth regions
        left_eye_pts = np.array([source_landmarks[i] for i in self.LEFT_EYE], dtype=np.int32)
        right_eye_pts = np.array([source_landmarks[i] for i in self.RIGHT_EYE], dtype=np.int32)
        lips_pts = np.array([source_landmarks[i] for i in self.LIPS], dtype=np.int32)
        
        cv2.fillPoly(eyes_mouth_mask, [left_eye_pts, right_eye_pts, lips_pts], (255, 255, 255))
        eyes_mouth_mask_gray = cv2.cvtColor(eyes_mouth_mask, cv2.COLOR_BGR2GRAY)
        
        # Create a version with more prominent eyes and mouth
        enhanced_parts = img_result.copy()
        enhanced_parts = cv2.addWeighted(enhanced_parts, 1.2, enhanced_parts, 0, 10)  # Increase contrast
        
        # Apply enhanced blending to eyes and mouth
        eyes_mask_3ch = cv2.cvtColor(eyes_mouth_mask_gray, cv2.COLOR_GRAY2BGR) / 255.0
        output = output * (1 - eyes_mask_3ch) + enhanced_parts * eyes_mask_3ch
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output
    
    def monitor_webcam(self):
        """Thread function to monitor webcam activity"""
        last_check = time.time()
        check_interval = 2  # Check every 2 seconds
        
        while self.running:
            current_time = time.time()
            
            # Check periodically if webcam is in use
            if current_time - last_check >= check_interval:
                last_check = current_time
                
                # Try to open the camera
                try:
                    if self.cap is None:
                        self.cap = cv2.VideoCapture(0)
                    
                    if not self.cap.isOpened():
                        self.cap.open(0)
                    
                    ret, _ = self.cap.read()
                    
                    if ret:
                        if not self.webcam_active:
                            print("Webcam is now active")
                            self.webcam_active = True
                            # Start face swapping
                            self.start_face_swap()
                    else:
                        if self.webcam_active:
                            print("Webcam is no longer active")
                            self.webcam_active = False
                            # Release resources
                            self.cap.release()
                            cv2.destroyAllWindows()
                            self.cap = None
                
                except Exception as e:
                    print(f"Error checking webcam status: {e}")
                    if self.cap:
                        self.cap.release()
                    self.cap = None
                    self.webcam_active = False
                    time.sleep(5)  # Wait longer before retrying after an error
            
            time.sleep(0.5)  # Short sleep to prevent CPU hogging
    
    def start_face_swap(self):
        """Start the face swapping process"""
        print("Starting face swap. Press 'q' to exit.")
        
        while self.webcam_active and self.running:
            try:
                # Capture frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    self.webcam_active = False
                    break
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]
                
                # Create a side-by-side view
                display_width = w * 2
                display_img = np.zeros((h, display_width, 3), dtype=np.uint8)
                display_img[:, :w] = frame  # Original frame on the left
                
                # Detect faces in the frame
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    # Get facial landmarks
                    landmarks = []
                    for landmark in results.multi_face_landmarks[0].landmark:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        landmarks.append((x, y))
                    
                    landmarks = np.array(landmarks)
                    
                    # Perform face swap
                    result = self.swap_faces(frame, landmarks)
                    
                    # Add to the display (result on the right)
                    display_img[:, w:] = result
                    
                    # Show labels
                    cv2.putText(display_img, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(display_img, "Face Swap", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Display the result
                    cv2.imshow('Face Swap', display_img)
                else:
                    # No face detected, show original frame on both sides
                    display_img[:, w:] = frame
                    cv2.putText(display_img, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(display_img, "No Face Detected", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow('Face Swap', display_img)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.webcam_active = False
                    break
                    
            except Exception as e:
                print(f"Error during face swap: {e}")
                import traceback
                traceback.print_exc()
                self.webcam_active = False
                break
                
        # Clean up resources
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def start_monitoring(self):
        """Start monitoring the webcam in a separate thread"""
        self.monitor_thread = threading.Thread(target=self.monitor_webcam)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Webcam monitoring started")
        
    def stop(self):
        """Stop all threads and release resources"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Face swapper stopped")


def main():
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the reference image (the face to swap with)
    # Try multiple possible locations
    possible_paths = [
        "reference_face2.jpg",  # Same directory
        os.path.join(current_dir, "reference_face2.jpg"),  # Absolute path
        os.path.join(os.getcwd(), "reference_face2.jpg"),  # Current working directory
        os.path.join(os.path.expanduser("~"), "reference_face2.jpg")  # Home directory
    ]
    
    # Try to find the reference image
    reference_image_path = None
    for path in possible_paths:
        if os.path.exists(path):
            reference_image_path = path
            break
    
    # If no reference image found, ask user for input
    if reference_image_path is None:
        print("Reference image 'reference_face2.jpg' not found.")
        reference_image_path = input("Please enter the full path to your reference image: ").strip()
        
        if not os.path.exists(reference_image_path):
            print(f"Error: File not found at '{reference_image_path}'")
            sys.exit(1)
    
    print("Starting Improved Face Swap Webcam Tool")
    print("------------------------------------")
    print(f"Reference image: {reference_image_path}")
    print("Press 'q' to quit when the face swap window is active")
    
    # Create and start the face swapper
    swapper = ImprovedFaceSwapper(reference_image_path)
    swapper.start_monitoring()
    
    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping face swapper...")
    finally:
        swapper.stop()
    
    print("Face swapper terminated")

if __name__ == "__main__":
    main()
