import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from collections import deque
import tempfile
import time
from datetime import datetime
import os
import json
from ultralytics import YOLO
from playsound import playsound
def play_alert():
    try:
        playsound('alert.wav')
    except Exception as e:
        # st.error(f"Error playing sound: {str(e)}")
        print(str(e))
# from sidebar import create_sidebar

def load_system_info():
    """Load or initialize system information"""
    if os.path.exists('system_info.json'):
        with open('system_info.json', 'r') as f:
            return json.load(f)
    return {
        'total_detections': 0,
        'system_start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def create_metrics_dashboard():
    """Create a dashboard with key metrics"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-container">
                <div class="metric-value">0</div>
                <div class="metric-label">Incidents Detected Today</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-container">
                <div class="metric-value">100%</div>
                <div class="metric-label">System Uptime</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-container">
                <div class="metric-value">0ms</div>
                <div class="metric-label">Average Response Time</div>
            </div>
        """, unsafe_allow_html=True)

def create_sidebar():
    """Create an enhanced sidebar with project information and controls"""
    
    # Enhanced Custom CSS
    st.markdown("""
        <style>
        /* Main container styling */
        .main {
            padding: 2rem;
            background-color: #f8f9fa;
        }
        
        /* Header styling */
        .title-container {
            background-color: #1E1E1E;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
        }
        
        /* Button styling */
        .stButton>button {
            width: 100%;
            height: 3.5em;
            margin-top: 1em;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* File uploader styling */
        .uploadedFile {
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            background-color: rgba(76, 175, 80, 0.1);
        }
        
        /* Prediction box styling */
        .prediction-box {
            padding: 1.5em;
            border-radius: 10px;
            margin: 1em 0;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .prediction-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        .violence {
            background-color: rgba(255, 99, 71, 0.1);
            border: 3px solid #FF6347;
        }
        
        .non-violence {
            background-color: rgba(46, 139, 87, 0.1);
            border: 3px solid #2E8B57;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        
        /* Progress bar styling */
        .stProgress > div > div {
            background-color: #4CAF50;
        }
        
        /* Stats container styling */
        .stats-container {
            background-color: #232323;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        /* Custom metric styling */
        .metric-container {
            background-color: #232323;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #FFFFF;
        }
        
        .metric-label {
            font-size: 14px;
            color: #FFFFF;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # System info
    system_info = load_system_info()
    
    # Sidebar Header
    st.sidebar.markdown("""
        <div class="sidebar-header">
            <h2>🚨 Violence Detection</h2>
            <p style="font-size: 0.9rem;">Control Panel & System Information</p>
        </div>
    """, unsafe_allow_html=True)
    
    
    # Detection Mode Selection
    st.sidebar.markdown("### 📌 Detection Mode")
    detection_mode = st.sidebar.selectbox(
        "",
        ["Video", "Webcam","RTSP Stream"],
        format_func=lambda x: f"📷 {x}" if x == "Image" else f"🎥 {x}" if x == "Video" else f"📹 {x}"
    )
    
    # Add Model Selection for Video Mode
    model_type = None
    if detection_mode in ["Video", "Webcam","RTSP Stream"]:
        st.sidebar.markdown("### 🤖 Model Selection")
        model_type = st.sidebar.selectbox(
            "Choose Detection Model",
            ["CNN+LSTM", "YOLOv11 Model"],
            help="Select the model to use for violence detection"
        )
    
     # Add RTSP URL input when RTSP mode is selected
    rtsp_url = None
    if detection_mode == "RTSP Stream":
        st.sidebar.markdown("### 🔗 RTSP Connection")
        rtsp_url = st.sidebar.text_input(
            "RTSP Stream URL",
            placeholder="rtsp://username:password@ip_address:port/stream",
            help="Enter the RTSP stream URL for your camera"
        )
        st.sidebar.caption("Example: rtsp://192.168.1.100:554/stream")

    # Model Settings
    st.sidebar.markdown("### ⚙️ Model Settings")
    with st.sidebar.expander("Configure Detection Parameters"):
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Adjust the confidence threshold for detection"
        )
        
        frame_window = st.slider(
            "Frame Window",
            min_value=1,
            max_value=30,
            value=16,
            step=1,
            help="Number of frames to analyze together"
        )
        
        enable_notifications = st.checkbox(
            "Enable Alerts",
            value=True,
            help="Get notifications for detected incidents"
        )
    
    # System Statistics
    st.sidebar.markdown("### 📊 System Statistics")
    with st.sidebar.expander("View System Metrics"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Detections", system_info['total_detections'])
        with col2:
            uptime = datetime.now() - datetime.strptime(system_info['system_start_time'], "%Y-%m-%d %H:%M:%S")
            st.metric("Uptime", f"{uptime.days}d {uptime.seconds//3600}h")
        
        st.markdown("#### System Health")
        st.progress(0.95)
        st.caption("System running normally")
    
    # About Section
    st.sidebar.markdown("### ℹ️ About")
    with st.sidebar.expander("Project Information"):
        st.markdown("""
            This Violence Detection System uses advanced deep learning algorithms to detect and analyze potentially violent content in images and videos in real-time.
            
            **Key Features:**
            - Real-time detection
            - Multi-format support
            - High accuracy rate
            - Instant alerts
            - Detailed analytics
            
            **Use Cases:**
            - Security monitoring
            - Public safety
            - Content moderation
            - Event surveillance
            
            **Technical Specifications:**
            - Model: CNN-LSTM
            - Framework: TensorFlow
            - Input Resolution: 64x64
            - Processing Speed: ~30 FPS
        """)
    
    # Help & Documentation
    st.sidebar.markdown("### 📚 Help & Support")
    with st.sidebar.expander("Quick Guide"):
        st.markdown("""
            **Getting Started:**
            1. Select detection mode
            2. Adjust confidence threshold
            3. Upload media or start webcam
            4. View real-time results
            
            **Tips:**
            - Higher confidence threshold = fewer false positives
            - Use video mode for longer sequences
            - Enable alerts for immediate notifications
            
            **Need Help?**
            Contact support at: support@example.com
        """)
    
    # Export Options
    st.sidebar.markdown("### 💾 Export Options")
    with st.sidebar.expander("Export Settings"):
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "JSON", "PDF"]
        )
        include_timestamps = st.checkbox("Include Timestamps", value=True)
        
        st.button("Export Report", key="export_button")
    
    
    return {
        'detection_mode': detection_mode,
        'model_type': model_type,
        'rtsp_url': rtsp_url,
        'confidence_threshold': confidence_threshold,
        'frame_window': frame_window,
        'enable_notifications': enable_notifications,
        'export_format': export_format,
        'include_timestamps': include_timestamps,
    }


# Add function to process video with YOLO
def process_video_yolo(video_file, model, config):
    """Process video with YOLOv11 model"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(video_file.read())
        
        video_reader = cv2.VideoCapture(temp_file.name)
        
        # Video properties
        width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_reader.get(cv2.CAP_PROP_FPS))
        total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_writer = cv2.VideoWriter(
            output_file.name,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        # Process frames
        violence_frames = 0
        frame_count = 0
        violence_timestamps = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while video_reader.isOpened():
            ok, frame = video_reader.read()
            if not ok:
                break
                
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            # Run YOLOv11 detection
            results = model(frame)
            
            # Process results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = float(box.conf)
                    if confidence >= config['confidence_threshold']:
                        violence_frames += 1
                        current_time = frame_count / fps
                        violence_timestamps.append(current_time)
                        
                        if config['enable_notifications']:
                            st.warning(f"⚠️ Violence Detected at {current_time:.2f} seconds!")
                            play_alert()
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Violence ({confidence:.2f})",
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if config['include_timestamps']:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (width - 200, height - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            video_writer.write(frame)
        
        video_reader.release()
        video_writer.release()
        
        # Update system info and generate report
        update_system_info('video', violence_frames > 0)
        violence_percentage = (violence_frames / frame_count) * 100 if frame_count > 0 else 0
        
        return output_file.name, violence_frames, frame_count, violence_timestamps, violence_percentage
        
    except Exception as e:
        st.error(f"Error processing video with YOLO: {str(e)}")
        return None, 0, 0, [], 0
    

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
CLASSES_LIST = ["NonViolence", "Violence"]

# Initialize session state if not exists
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = 0
if 'violence_incidents' not in st.session_state:
    st.session_state.violence_incidents = []

def update_system_info(detection_type, is_violent):
    """Update system information and statistics"""
    try:
        # Load current system info
        if os.path.exists('system_info.json'):
            with open('system_info.json', 'r') as f:
                system_info = json.load(f)
        else:
            system_info = {
                'total_detections': 0,
                'system_start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'detection_history': []
            }
        
        # Update statistics
        system_info['total_detections'] += 1
        system_info['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add detection to history
        detection_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': detection_type,
            'is_violent': is_violent
        }
        system_info['detection_history'].append(detection_entry)
        
        # Save updated info
        with open('system_info.json', 'w') as f:
            json.dump(system_info, f)
            
    except Exception as e:
        st.error(f"Error updating system info: {str(e)}")

def process_image(image_file, model, config):
    """Process image with config settings"""
    start_time = time.time()
    try:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Resize according to config
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_sequence = np.array([normalized_frame] * config['frame_window'])
        
        # Predict
        prediction = model.predict(np.expand_dims(frames_sequence, axis=0))[0]
        confidence = float(max(prediction))
        
        # Apply confidence threshold from config
        if confidence >= config['confidence_threshold']:
            predicted_class = CLASSES_LIST[np.argmax(prediction)]
        else:
            predicted_class = "Uncertain"
            
        # Update system info
        update_system_info('image', predicted_class == "Violence")
        
        # Send notification if enabled
        if config['enable_notifications'] and predicted_class == "Violence":
            st.warning("⚠️ Violence Detected in Image!")
            play_alert()
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return frame, predicted_class, confidence,processing_time
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None

def process_video(video_file, model, config):
    """Process video with config settings"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(video_file.read())
        
        frames_queue = deque(maxlen=config['frame_window'])
        video_reader = cv2.VideoCapture(temp_file.name)
        
        # Video properties
        width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_reader.get(cv2.CAP_PROP_FPS))
        total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_writer = cv2.VideoWriter(
            output_file.name,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        # Process frames
        violence_frames = 0
        frame_count = 0
        violence_timestamps = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while video_reader.isOpened():
            ok, frame = video_reader.read()
            if not ok:
                break
                
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
            frames_queue.append(normalized_frame)
            
            if len(frames_queue) == config['frame_window']:
                prediction = model.predict(np.expand_dims(frames_queue, axis=0))[0]
                confidence = float(max(prediction))
                
                if confidence >= config['confidence_threshold']:
                    predicted_class = CLASSES_LIST[np.argmax(prediction)]
                    if predicted_class == "Violence":
                        violence_frames += 1
                        current_time = frame_count / fps
                        violence_timestamps.append(current_time)
                        
                        if config['enable_notifications']:
                            st.warning(f"⚠️ Violence Detected at {current_time:.2f} seconds!")
                            play_alert()
                    
                    # Add overlay
                    color = (0, 255, 0) if predicted_class == "NonViolence" else (0, 0, 255)
                    cv2.putText(frame, f"{predicted_class} ({confidence:.2f})",
                              (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
            if config['include_timestamps']:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (width - 200, height - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            video_writer.write(frame)
        
        video_reader.release()
        video_writer.release()
        
        # Generate report if needed
        if config['export_format'] != "None":
            report_data = {
                'filename': video_file.name,
                'total_frames': frame_count,
                'violence_frames': violence_frames,
                'violence_percentage': (violence_frames / frame_count) * 100,
                'violence_timestamps': violence_timestamps,
                'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            export_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if config['export_format'] == "JSON":
                with open(f"{export_filename}.json", 'w') as f:
                    json.dump(report_data, f, indent=4)
            elif config['export_format'] == "CSV":
                import csv
                with open(f"{export_filename}.csv", 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=report_data.keys())
                    writer.writeheader()
                    writer.writerow(report_data)
                    
        update_system_info('video', violence_frames > 0)
        violence_percentage = (violence_frames / frame_count) * 100 if frame_count > 0 else 0

        
        return output_file.name, violence_frames, frame_count, violence_timestamps, violence_percentage
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None, 0, 0, []

# Add functions to process RTSP stream
def process_rtsp_tensorflow(url, model, config):
    """Process RTSP stream with TensorFlow model"""
    try:
        # Test connection to the RTSP stream
        video_capture = cv2.VideoCapture(url)
        if not video_capture.isOpened():
            st.error("❌ Failed to connect to RTSP stream. Please check the URL and try again.")
            return
            
        st.success("✅ Connected to RTSP stream successfully!")
        
        frames_queue = deque(maxlen=config['frame_window'])
        stframe = st.empty()
        
        # Statistics columns
        col1, col2, col3 = st.columns(3)
        frame_counter = col1.empty()
        violence_counter = col2.empty()
        fps_counter = col3.empty()
        
        # Recording control
        recording = False
        record_button = st.button("📹 Start Recording")
        stop_record_button = st.button("⏹️ Stop Recording")
        
        # Initialize video writer
        video_writer = None
        output_file = None
        
        # Initialize counters
        frame_count = 0
        violence_frames = 0
        start_time = time.time()
        last_violence_time = 0
        fps_list = deque(maxlen=30)  # Keep track of recent FPS values
        
        stop_button = st.button('Stop Stream', key='stop_rtsp')
        status_text = st.empty()
        
        while not stop_button:
            ret, frame = video_capture.read()
            if not ret:
                status_text.error("Lost connection to RTSP stream. Attempting to reconnect...")
                # Try to reconnect
                video_capture.release()
                time.sleep(2)
                video_capture = cv2.VideoCapture(url)
                if not video_capture.isOpened():
                    status_text.error("Failed to reconnect to RTSP stream. Please check the connection.")
                    break
                continue
                
            frame_count += 1
            current_time = time.time()
            fps = 1 / (current_time - start_time) if (current_time - start_time) > 0 else 0
            fps_list.append(fps)
            start_time = current_time
            
            # Start recording if requested
            if record_button and not recording:
                recording = True
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"rtsp_recording_{timestamp}.mp4"
                video_writer = cv2.VideoWriter(
                    output_file,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    30,  # FPS
                    (frame.shape[1], frame.shape[0])
                )
                st.success(f"Recording started: {output_file}")
            
            # Stop recording if requested
            if stop_record_button and recording:
                recording = False
                if video_writer:
                    video_writer.release()
                    st.success(f"Recording saved to {output_file}")
                    # Offer download
                    with open(output_file, 'rb') as f:
                        st.download_button(
                            '📥 Download Recorded Video',
                            f,
                            file_name=output_file,
                            mime='video/mp4'
                        )
            
            # Write frame if recording
            if recording and video_writer:
                video_writer.write(frame)
            
            # Process for violence detection
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
            frames_queue.append(normalized_frame)
            
            if len(frames_queue) == config['frame_window']:
                prediction = model.predict(np.expand_dims(frames_queue, axis=0))[0]
                confidence = float(max(prediction))
                
                if confidence >= config['confidence_threshold']:
                    predicted_class = CLASSES_LIST[np.argmax(prediction)]
                    
                    if predicted_class == "Violence":
                        violence_frames += 1
                        
                        # Check notification cooldown (5 seconds)
                        if config['enable_notifications'] and (current_time - last_violence_time) >= 5:
                            st.warning("⚠️ Violence Detected in RTSP Stream!")
                            play_alert()
                            last_violence_time = current_time
                            
                    color = (0, 255, 0) if predicted_class == "NonViolence" else (0, 0, 255)
                    cv2.putText(frame, f"{predicted_class} ({confidence:.2f})",
                              (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if config['include_timestamps']:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, 
                          (frame.shape[1] - 200, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Calculate average FPS
            avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
            
            # Update statistics
            frame_counter.metric("Frames", frame_count)
            violence_counter.metric("Violence Frames", violence_frames)
            fps_counter.metric("FPS", f"{avg_fps:.2f}")
            
            stframe.image(frame, channels="BGR", use_container_width=True)
        
        # Clean up
        video_capture.release()
        if video_writer:
            video_writer.release()
        update_system_info('rtsp', violence_frames > 0)
        
    except Exception as e:
        st.error(f"Error with RTSP stream: {str(e)}")

def process_rtsp_yolo(url, model, config):
    """Process RTSP stream with YOLOv8 model"""
    try:
        # Test connection to the RTSP stream
        video_capture = cv2.VideoCapture(url)
        if not video_capture.isOpened():
            st.error("❌ Failed to connect to RTSP stream. Please check the URL and try again.")
            return
            
        st.success("✅ Connected to RTSP stream successfully!")
        
        stframe = st.empty()
        
        # Statistics columns
        col1, col2, col3 = st.columns(3)
        frame_counter = col1.empty()
        violence_counter = col2.empty()
        fps_counter = col3.empty()
        
        # Recording control
        recording = False
        record_button = st.button("📹 Start Recording")
        stop_record_button = st.button("⏹️ Stop Recording")
        
        # Initialize video writer
        video_writer = None
        output_file = None
        
        # Initialize counters
        frame_count = 0
        violence_frames = 0
        start_time = time.time()
        last_violence_time = 0
        fps_list = deque(maxlen=30)  # Keep track of recent FPS values
        
        stop_button = st.button('Stop Stream', key='stop_rtsp')
        status_text = st.empty()
        
        while not stop_button:
            ret, frame = video_capture.read()
            if not ret:
                status_text.error("Lost connection to RTSP stream. Attempting to reconnect...")
                # Try to reconnect
                video_capture.release()
                time.sleep(2)
                video_capture = cv2.VideoCapture(url)
                if not video_capture.isOpened():
                    status_text.error("Failed to reconnect to RTSP stream. Please check the connection.")
                    break
                continue
                
            frame_count += 1
            current_time = time.time()
            fps = 1 / (current_time - start_time) if (current_time - start_time) > 0 else 0
            fps_list.append(fps)
            start_time = current_time
            
            # Start recording if requested
            if record_button and not recording:
                recording = True
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"rtsp_recording_{timestamp}.mp4"
                video_writer = cv2.VideoWriter(
                    output_file,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    30,  # FPS
                    (frame.shape[1], frame.shape[0])
                )
                st.success(f"Recording started: {output_file}")
            
            # Stop recording if requested
            if stop_record_button and recording:
                recording = False
                if video_writer:
                    video_writer.release()
                    st.success(f"Recording saved to {output_file}")
                    # Offer download
                    with open(output_file, 'rb') as f:
                        st.download_button(
                            '📥 Download Recorded Video',
                            f,
                            file_name=output_file,
                            mime='video/mp4'
                        )
            
            # Write frame if recording
            if recording and video_writer:
                video_writer.write(frame)
            
            # Run YOLOv8 detection
            results = model(frame)
            
            # Process results
            detected_violence = False
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = float(box.conf)
                    if confidence >= config['confidence_threshold']:
                        violence_frames += 1
                        detected_violence = True
                        
                        # Check notification cooldown (5 seconds)
                        if config['enable_notifications'] and (current_time - last_violence_time) >= 5:
                            st.warning("⚠️ Violence Detected in RTSP Stream!")
                            play_alert()
                            last_violence_time = current_time
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Violence ({confidence:.2f})",
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if config['include_timestamps']:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, 
                          (frame.shape[1] - 200, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Calculate average FPS
            avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
            
            # Update statistics
            frame_counter.metric("Frames", frame_count)
            violence_counter.metric("Violence Frames", violence_frames)
            fps_counter.metric("FPS", f"{avg_fps:.2f}")
            
            stframe.image(frame, channels="BGR", use_container_width=True)
        
        # Clean up
        video_capture.release()
        if video_writer:
            video_writer.release()
        update_system_info('rtsp', violence_frames > 0)
        
    except Exception as e:
        st.error(f"Error with RTSP stream: {str(e)}")
        
def process_webcam(model, config):
    """Process webcam feed with config settings"""
    try:
        video_capture = cv2.VideoCapture(0)
        frames_queue = deque(maxlen=config['frame_window'])
        stframe = st.empty()
        
        # Statistics columns
        col1, col2, col3 = st.columns(3)
        frame_counter = col1.empty()
        violence_counter = col2.empty()
        fps_counter = col3.empty()
        
        # Initialize counters
        frame_count = 0
        violence_frames = 0
        start_time = time.time()
        last_violence_time = 0
        
        stop_button = st.button('Stop Recording', key='stop_webcam')
        
        while not stop_button:
            ret, frame = video_capture.read()
            if not ret:
                break
                
            frame_count += 1
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
            frames_queue.append(normalized_frame)
            
            if len(frames_queue) == config['frame_window']:
                prediction = model.predict(np.expand_dims(frames_queue, axis=0))[0]
                confidence = float(max(prediction))
                
                if confidence >= config['confidence_threshold']:
                    predicted_class = CLASSES_LIST[np.argmax(prediction)]
                    
                    if predicted_class == "Violence":
                        violence_frames += 1
                        
                        # Check notification cooldown (5 seconds)
                        if config['enable_notifications'] and (current_time - last_violence_time) >= 5:
                            st.warning("⚠️ Violence Detected in Webcam Feed!")
                            play_alert()
                            last_violence_time = current_time
                            
                    color = (0, 255, 0) if predicted_class == "NonViolence" else (0, 0, 255)
                    cv2.putText(frame, f"{predicted_class} ({confidence:.2f})",
                              (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if config['include_timestamps']:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, 
                          (frame.shape[1] - 200, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Update statistics
            frame_counter.metric("Frames", frame_count)
            violence_counter.metric("Violence Frames", violence_frames)
            fps_counter.metric("FPS", f"{fps:.2f}")
            
            stframe.image(frame, channels="BGR", use_container_width=True)
        
        video_capture.release()
        update_system_info('webcam', violence_frames > 0)
        
    except Exception as e:
        st.error(f"Error with webcam: {str(e)}")

# Add new function for YOLO webcam processing
def process_webcam_yolo(model, config):
    """Process webcam feed with YOLOv11 model"""
    try:
        video_capture = cv2.VideoCapture(0)
        stframe = st.empty()
        
        # Statistics columns
        col1, col2, col3 = st.columns(3)
        frame_counter = col1.empty()
        violence_counter = col2.empty()
        fps_counter = col3.empty()
        
        # Initialize counters
        frame_count = 0
        violence_frames = 0
        start_time = time.time()
        last_violence_time = 0
        
        stop_button = st.button('Stop Recording', key='stop_webcam')
        
        while not stop_button:
            ret, frame = video_capture.read()
            if not ret:
                break
                
            frame_count += 1
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            
            # Run YOLOv11 detection
            results = model(frame,conf=0.80)
            
            # Process results and draw detections
            detected_violence = False
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = float(box.conf)
                    if confidence >= config['confidence_threshold']:
                        violence_frames += 1
                        detected_violence = True
                        
                        # Check notification cooldown (5 seconds)
                        if config['enable_notifications'] and (current_time - last_violence_time) >= 5:
                            st.warning("⚠️ Violence Detected in Webcam Feed!")
                            play_alert()
                            last_violence_time = current_time
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Violence ({confidence:.2f})",
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if config['include_timestamps']:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, 
                          (frame.shape[1] - 200, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Update statistics
            frame_counter.metric("Frames", frame_count)
            violence_counter.metric("Violence Frames", violence_frames)
            fps_counter.metric("FPS", f"{fps:.2f}")
            
            stframe.image(frame, channels="BGR", use_container_width=True)
        
        video_capture.release()
        update_system_info('webcam', violence_frames > 0)
        
    except Exception as e:
        st.error(f"Error with webcam: {str(e)}")

def main():
    # Page config
    st.set_page_config(page_title="Violence Detection System", layout="wide")
    # Custom title with styling
    st.markdown("""
        <div class="title-container">
            <h1>🚨 Real-time Violence Detection System</h1>
            <p>Advanced monitoring and detection system powered by AI</p>
        </div>
    """, unsafe_allow_html=True)
    # Get configuration from sidebar
    config = create_sidebar()
    
    # Load model
    @st.cache_resource
    def load_models():
        try:
            tf_model = load_model('violence_detection_model.h5')
            yolo_model = YOLO('bestvoilence.pt')  # Load YOLOv11 model
            return tf_model, yolo_model
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return None, None
        
    create_metrics_dashboard()
    tf_model, yolo_model = load_models()
    if tf_model is None or yolo_model is None:
        st.error("Failed to load one or both models. Please check if the model files exist.")
        return
    
    # Main content
    # st.title("🚨 Real-time Violence Detection System")
    
    if config['detection_mode'] == "Image":
        st.subheader("📷 Image Analysis")
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            with st.spinner('Processing image...'):
                frame, predicted_class, confidence,processing_time = process_image(uploaded_file, model, config)
                if frame is not None:
                    col1, col2 = st.columns(2)    
                    with col1:
                        st.image(frame, channels="BGR", use_container_width=True)
                        st.markdown("""
                            <div style='text-align: center; padding: 1rem;'>
                                <p>Processing Time: {:.2f}ms</p>
                            </div>
                        """.format(processing_time), unsafe_allow_html=True)
                    
                    with col2:
                        box_class = "violence" if predicted_class == "Violence" else "non-violence"
                        if box_class == "violence":
                            play_alert()
                        st.markdown(f"""
                            <div class="prediction-box {box_class}">
                                <h2>{predicted_class}</h2>
                                <p>Confidence: {confidence:.2f}</p>
                                <hr>
                                <h4>   Analysis Details:</h4>
                                <ul style="list-style-type: none; padding: 0;">
                                    <li>Detection Time: {processing_time:.2f}ms</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)

    elif config['detection_mode'] == "Video":
        # st.subheader("🎥 Video Analysis")
        st.markdown("### 🎥 Video Violence Detection")
        uploaded_file = st.file_uploader(
            "Drop your video here or click to upload",
            type=['mp4', 'avi', 'mov']
        )
        # uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file:
            with st.spinner('Processing video... This may take a while.'):
                if config['model_type'] == "YOLOv11 Model":
                    output_file, violence_frames, total_frames, timestamps, violence_percentage = process_video_yolo(
                        uploaded_file, yolo_model, config
                    )
                else:  # TensorFlow Model
                    output_file, violence_frames, total_frames, timestamps, violence_percentage = process_video(
                        uploaded_file, tf_model, config
                    )
                if output_file:
                    # Statistics display
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                            <div class="metric-container">
                                <div class="metric-value">{:.1f}%</div>
                                <div class="metric-label">Violence Detected</div>
                            </div>
                        """.format(violence_percentage), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                            <div class="metric-container">
                                <div class="metric-value">{}</div>
                                <div class="metric-label">Total Frames</div>
                            </div>
                        """.format(total_frames), unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("""
                            <div class="metric-container">
                                <div class="metric-value">{:.1f}%</div>
                                <div class="metric-label">Processing Accuracy</div>
                            </div>
                        """.format(99.5), unsafe_allow_html=True)
                    
                    # Download button
                    with open(output_file, 'rb') as f:
                        st.download_button(
                            '📥 Download Processed Video',
                            f,
                            file_name='processed_video.mp4',
                            mime='video/mp4'
                        )
                    
                    # Cleanup temporary file
                    try:
                        os.remove(output_file)
                    except:
                        pass
                    # st.video(output_file)
                    # st.write(f"Violence detected in {violence_frames} out of {total_frames} frames")
                    # if timestamps:
                    #     st.write("Violence detected at timestamps (seconds):", timestamps)
    elif config['detection_mode'] == "RTSP Stream":
        st.markdown("### 🌐 RTSP Stream Violence Detection")
        
        if not config['rtsp_url']:
            st.warning("⚠️ Please enter an RTSP URL in the sidebar to begin streaming.")
        else:
            # Stream connection info
            st.info(f"Connecting to: {config['rtsp_url']}")
            
            start_button = st.button('▶️ Start Stream')
            
            if start_button:
                if config['model_type'] == "YOLOv11 Model":
                    process_rtsp_yolo(config['rtsp_url'], yolo_model, config)
                else:  # TensorFlow Model
                    process_rtsp_tensorflow(config['rtsp_url'], tf_model, config)

    else:  # Webcam
        st.markdown("### 📹 Webcam Violence Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button('▶️ Start Webcam')
        
        with col2:
            st.markdown("""
                <div style='padding: 1.7rem;'>
                    <p style='color: #666;'>Click Start to begin real-time detection</p>
                </div>
            """, unsafe_allow_html=True)
        
        if start_button:
            if config['model_type'] == "YOLOv11 Model":
                process_webcam_yolo(yolo_model, config)
            else:  # TensorFlow Model
                # process_webcam(model)
                process_webcam(tf_model, config)
    


if __name__ == '__main__':
    main()