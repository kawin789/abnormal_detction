import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import pygame
from collections import deque
import os
import hashlib

# Initialize session state for login tracking
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['current_tab'] = "Upload Video"

if 'username' not in st.session_state:
    st.session_state['username'] = None

# Initialize session state for live analytics data
if 'live_data' not in st.session_state:
    st.session_state['live_data'] = {
        'timestamps': [],
        'normal_count': [],
        'anomaly_count': []
    }

# Database handling for users
def init_user_db():
    conn = sqlite3.connect('user_accounts.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    # Create default admin user if not exists
    cursor.execute('SELECT * FROM users WHERE username = ?', ('admin',))
    if not cursor.fetchone():
        # Default password: admin123
        password_hash = hashlib.sha256('admin123'.encode()).hexdigest()
        cursor.execute('INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)', 
                       ('admin', password_hash, 'admin'))
    conn.commit()
    return conn, cursor

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    conn, cursor = init_user_db()
    cursor.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()
    
    if result and result[0] == hash_password(password):
        return True
    return False

def register_user(username, password, role='user'):
    conn, cursor = init_user_db()
    try:
        cursor.execute('INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)', 
                   (username, hash_password(password), role))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def login_page():
    st.title("Anomaly Detection System")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            if verify_user(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success("Login successful!")
                st.rerun()  # Changed from experimental_rerun
            else:
                st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Register")
        new_username = st.text_input("Username", key="reg_username")
        new_password = st.text_input("Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
        
        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
            elif not new_username or not new_password:
                st.error("Username and password cannot be empty")
            else:
                if register_user(new_username, new_password, "user"):
                    st.success(f"User {new_username} registered successfully! You can now login.")
                else:
                    st.error(f"Username '{new_username}' already exists")

class AnomalyDetector:
    def __init__(self):
        # Setup output and database
        self.output_dir = Path("anomaly_captures")
        self.output_dir.mkdir(exist_ok=True)
        
        self.image_dir = self.output_dir / "incident_captures"
        self.image_dir.mkdir(exist_ok=True)
        
        # Database setup
        self.conn = sqlite3.connect('anomaly_analytics.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS behavior_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT,
                timestamp TEXT,
                situation TEXT,
                before_image TEXT,
                during_image TEXT,
                after_image TEXT,
                incident_type TEXT
            )
        ''')
        self.conn.commit()

        # Model setup - with error handling
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
            self.model.classes = [0]  # People detection
            self.model.conf = 0.5
        except RuntimeError as e:
            st.error(f"Error loading YOLOv5 model: {str(e)}")
            st.info("Try clearing the torch cache or restarting the application")
            self.model = None

        # Sound setup
        try:
            pygame.mixer.init()
            self.alarm_sound = pygame.mixer.Sound('alarm.wav')
        except:
            st.warning("Could not initialize sound. Will continue without audio alerts.")
            self.alarm_sound = None
        
        # Frame buffer for situation capture
        self.frame_buffer = deque(maxlen=5)  # Store last 5 frames
        
        # Analytics tracking
        self.normal_frames = 0
        self.anomaly_frames = 0
        self.start_time = datetime.now()
        
    def create_loading_animation(self):
        """Create a custom loading animation"""
        with st.spinner("Processing video..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            return progress_bar, status_text

    def create_live_charts(self):
        """Initialize live charts"""
        col1, col2 = st.columns(2)
        with col1:
            pie_placeholder = st.empty()
        with col2:
            line_placeholder = st.empty()
        return pie_placeholder, line_placeholder

    def update_charts(self, pie_placeholder, line_placeholder):
        """Update charts with current statistics"""
        # Update pie chart
        pie_data = {
            'Behavior': ['Normal', 'Anomaly'],
            'Frames': [self.normal_frames, self.anomaly_frames]
        }
        fig_pie = px.pie(pd.DataFrame(pie_data), 
                    values='Frames', 
                    names='Behavior',
                    title='Behavior Distribution',
                    color_discrete_sequence=['#2ecc71', '#e74c3c'])
        pie_placeholder.plotly_chart(fig_pie, use_container_width=True)
        
        # Update line chart
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds()
        
        # Only add data points every second to avoid overcrowding
        if not st.session_state['live_data']['timestamps'] or elapsed - st.session_state['live_data']['timestamps'][-1] >= 1:
            st.session_state['live_data']['timestamps'].append(elapsed)
            st.session_state['live_data']['normal_count'].append(self.normal_frames)
            st.session_state['live_data']['anomaly_count'].append(self.anomaly_frames)
        
        df_line = pd.DataFrame({
            'Time (s)': st.session_state['live_data']['timestamps'],
            'Normal Frames': st.session_state['live_data']['normal_count'],
            'Anomaly Frames': st.session_state['live_data']['anomaly_count']
        })
        
        fig_line = px.line(df_line, x='Time (s)', y=['Normal Frames', 'Anomaly Frames'],
                      title='Live Detection Counts',
                      color_discrete_map={'Normal Frames': '#2ecc71', 'Anomaly Frames': '#e74c3c'})
        line_placeholder.plotly_chart(fig_line, use_container_width=True)

    def capture_situation(self, current_frame, incident_type):
        """Capture before, during, and after frames of an incident"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get frames from buffer for 'before' shots
        before_frames = list(self.frame_buffer)
        
        # Save before images (last 2 frames from buffer)
        before_path = str(self.image_dir / f"before_{timestamp}.jpg")
        if before_frames:
            cv2.imwrite(before_path, before_frames[-1])
        
        # Save during image (current frame)
        during_path = str(self.image_dir / f"during_{timestamp}.jpg")
        cv2.imwrite(during_path, current_frame)
        
        # Return paths and timestamp for database
        return timestamp, before_path, during_path

    def process_video(self, video_path, video_name):
        if self.model is None:
            st.error("YOLOv5 model failed to load. Cannot process video.")
            return
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error: Could not open video file {video_path}")
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize UI elements
        progress_bar, status_text = self.create_loading_animation()
        pie_placeholder, line_placeholder = self.create_live_charts()
        
        # Video display
        st.markdown("<style>.center { display: flex; justify-content: center; }</style>", 
                  unsafe_allow_html=True)
        frame_placeholder = st.empty()
        
        frame_count = 0
        current_incident = None
        after_incident_counter = 0
        
        # Track processed incidents to prevent duplicates
        processed_incidents = set()

        # Reset counters and timestamp for new video
        self.normal_frames = 0
        self.anomaly_frames = 0
        self.start_time = datetime.now()
        
        # Reset live data for new video
        st.session_state['live_data'] = {
            'timestamps': [],
            'normal_count': [],
            'anomaly_count': []
        }
        
        # Add a stop button to manually stop processing
        stop_col = st.columns([1, 1, 1])
        with stop_col[1]:
            stop_button = st.button("Stop Processing")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Handle end of video
                st.info("End of video reached")
                break
                
            # Check if stop button was pressed
            if stop_button:
                st.info("Processing stopped by user")
                break

            # Update progress
            progress = min(99, int((frame_count / total_frames) * 100))
            progress_bar.progress(progress)
            
            # Performance optimization
            frame_count += 1
            if frame_count % 3 != 0:
                continue

            # Add frame to buffer
            self.frame_buffer.append(frame.copy())

            # Detect people and analyze
            try:
                results = self.model(frame)
                detections = results.pandas().xyxy[0]
                
                # Check for anomaly
                anomaly_type = self._check_anomaly(detections)
                is_anomaly = anomaly_type != "normal"
                
                if is_anomaly:
                    self.anomaly_frames += 1
                    box_color = (0, 0, 255)  # Red
                    status_text.write(f"⚠️ Detecting: {anomaly_type}")
                    
                    # New incident detection
                    if current_incident is None:
                        timestamp, before_path, during_path = self.capture_situation(
                            frame, anomaly_type)
                        
                        # Check if this incident looks similar to one we've already processed
                        # Simplest approach: check if we're near the end of the video
                        near_end = frame_count > (total_frames * 0.9)
                        
                        # Create a simple signature for the incident
                        people_positions = []
                        for _, detection in detections.iterrows():
                            center_x = (detection['xmin'] + detection['xmax']) / 2
                            center_y = (detection['ymin'] + detection['ymax']) / 2
                            people_positions.append((center_x, center_y))
                        
                        # Sort and round to create a stable signature
                        people_positions.sort()
                        incident_signature = tuple((round(x), round(y)) for x, y in people_positions)
                        
                        # Only process if we haven't seen this incident before
                        if incident_signature not in processed_incidents:
                            processed_incidents.add(incident_signature)
                            
                            current_incident = {
                                'timestamp': timestamp,
                                'type': anomaly_type,
                                'before_path': before_path,
                                'during_path': during_path
                            }
                            if self.alarm_sound:
                                self.alarm_sound.play()
                else:
                    self.normal_frames += 1
                    box_color = (0, 255, 0)  # Green
                    status_text.write("✅ Normal Activity")
                    
                    # Capture after-incident frames
                    if current_incident is not None:
                        after_incident_counter += 1
                        if after_incident_counter >= 5:  # Wait 5 frames after incident
                            after_path = str(self.image_dir / 
                                           f"after_{current_incident['timestamp']}.jpg")
                            cv2.imwrite(after_path, frame)
                            
                            # Save to database
                            self._save_incident(
                                video_name,
                                current_incident['timestamp'],
                                current_incident['type'],
                                current_incident['before_path'],
                                current_incident['during_path'],
                                after_path
                            )
                            
                            current_incident = None
                            after_incident_counter = 0

                # Annotate frame
                for _, detection in detections.iterrows():
                    cv2.rectangle(
                        frame, 
                        (int(detection['xmin']), int(detection['ymin'])),
                        (int(detection['xmax']), int(detection['ymax'])),
                        box_color, 2
                    )
                
                # Add status banner
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
                cv2.putText(frame, 
                           "ANOMALY DETECTED" if is_anomaly else "NORMAL ACTIVITY",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

                # Update display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb)
                
                # Update live charts (every 10 frames to avoid slowdowns)
                if frame_count % 10 == 0:
                    self.update_charts(pie_placeholder, line_placeholder)
            
            except Exception as e:
                st.error(f"Error processing frame {frame_count}: {str(e)}")
                break

        cap.release()
        progress_bar.progress(100)
        status_text.write("✅ Processing Complete!")
        
        # Do one final update of the charts
        self.update_charts(pie_placeholder, line_placeholder)
        
        # Show final report after processing
        self.generate_final_report()

    def _check_anomaly(self, detections):
        """Check for anomalous behavior"""
        people_count = len(detections)
        if people_count >= 2:
            boxes = detections[['xmin', 'ymin', 'xmax', 'ymax']].values
            centers = [(box[0] + box[2])/2 for box in boxes]
            
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    if abs(centers[i] - centers[j]) < 50:
                        return "close_proximity"
                        
        return "normal"

    def _save_incident(self, video_name, timestamp, incident_type, 
                      before_path, during_path, after_path):
        """Save incident details to database"""
        self.cursor.execute('''
            INSERT INTO behavior_analytics 
            (video_name, timestamp, incident_type, before_image, 
             during_image, after_image)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (video_name, timestamp, incident_type, before_path, 
              during_path, after_path))
        self.conn.commit()

    def generate_final_report(self):
        """Generate final analysis report"""
        st.subheader("Incident Report")
        
        # Get all incidents from database
        df = pd.read_sql_query("""
            SELECT timestamp, incident_type, 
                   before_image, during_image, after_image 
            FROM behavior_analytics 
            ORDER BY timestamp DESC
            LIMIT 5
        """, self.conn)
        
        if df.empty:
            st.info("No incidents detected in this video.")
            return
            
        # Display incidents with images
        for _, incident in df.iterrows():
            st.write(f"**Incident Time:** {incident['timestamp']}")
            st.write(f"**Type:** {incident['incident_type']}")
            
            cols = st.columns(3)
            with cols[0]:
                st.write("Before")
                if os.path.exists(incident['before_image']):
                    st.image(incident['before_image'])
                else:
                    st.write("Image not found")
            with cols[1]:
                st.write("During")
                if os.path.exists(incident['during_image']):
                    st.image(incident['during_image'])
                else:
                    st.write("Image not found")
            with cols[2]:
                st.write("After")
                if os.path.exists(incident['after_image']):
                    st.image(incident['after_image'])
                else:
                    st.write("Image not found")
            
            st.markdown("---")
        
        st.info("Showing the 5 most recent incidents. Go to History tab to see all incidents.")

def history_page():
    st.subheader("Incident History")
    
    # Get all incidents from database
    try:
        conn = sqlite3.connect('anomaly_analytics.db')
        df = pd.read_sql_query("""
            SELECT id, video_name, timestamp, incident_type 
            FROM behavior_analytics 
            ORDER BY timestamp DESC
        """, conn)
        conn.close()
        
        if df.empty:
            st.info("No incidents have been detected yet.")
            return
            
        # Add a filter for video name
        all_videos = ['All'] + df['video_name'].unique().tolist()
        selected_video = st.selectbox("Filter by video:", all_videos)
        
        if selected_video != 'All':
            filtered_df = df[df['video_name'] == selected_video]
        else:
            filtered_df = df
        
        # Display as table
        st.write(f"Showing {len(filtered_df)} incidents")
        st.dataframe(filtered_df)
        
        # Option to view incident details
        selected_id = st.selectbox("Select incident ID to view details:", 
                                  filtered_df['id'].tolist())
        
        if st.button("View Incident Details"):
            conn = sqlite3.connect('anomaly_analytics.db')
            incident = pd.read_sql_query(f"""
                SELECT * FROM behavior_analytics 
                WHERE id = {selected_id}
            """, conn).iloc[0]
            conn.close()
            
            st.write(f"**Incident ID:** {incident['id']}")
            st.write(f"**Video:** {incident['video_name']}")
            st.write(f"**Timestamp:** {incident['timestamp']}")
            st.write(f"**Type:** {incident['incident_type']}")
            
            cols = st.columns(3)
            with cols[0]:
                st.write("Before")
                if os.path.exists(incident['before_image']):
                    st.image(incident['before_image'])
                else:
                    st.write("Image not found")
            with cols[1]:
                st.write("During")
                if os.path.exists(incident['during_image']):
                    st.image(incident['during_image'])
                else:
                    st.write("Image not found")
            with cols[2]:
                st.write("After")
                if os.path.exists(incident['after_image']):
                    st.image(incident['after_image'])
                else:
                    st.write("Image not found")
    
    except Exception as e:
        st.error(f"Error accessing database: {str(e)}")

def analytics_dashboard():
    st.subheader("Analytics Dashboard")
    
    try:
        conn = sqlite3.connect('anomaly_analytics.db')
        df = pd.read_sql_query("""
            SELECT * FROM behavior_analytics
        """, conn)
        conn.close()
        
        if df.empty:
            st.info("No data available for analytics. Process videos to generate data.")
            return
        
        # Add timestamp as datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S')
        
        # Analytics cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Incidents", len(df))
        with col2:
            st.metric("Total Videos Analyzed", df['video_name'].nunique())
        with col3:
            if not df.empty:
                st.metric("Latest Incident", df['datetime'].max().strftime('%Y-%m-%d %H:%M:%S'))
            else:
                st.metric("Latest Incident", "None")
        
        # Incidents by type
        st.subheader("Incidents by Type")
        type_counts = df['incident_type'].value_counts().reset_index()
        type_counts.columns = ['Incident Type', 'Count']
        
        fig_type = px.bar(type_counts, x='Incident Type', y='Count', 
                        color='Incident Type', title='Incidents by Type')
        st.plotly_chart(fig_type, use_container_width=True)
        
        # Incidents over time
        st.subheader("Incidents over Time")
        df['date'] = df['datetime'].dt.date
        time_counts = df.groupby('date').size().reset_index()
        time_counts.columns = ['Date', 'Count']
        
        fig_time = px.line(time_counts, x='Date', y='Count', 
                         markers=True, title='Incidents per Day')
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Incidents by video
        st.subheader("Incidents by Video")
        video_counts = df['video_name'].value_counts().reset_index()
        video_counts.columns = ['Video', 'Count']
        
        fig_video = px.pie(video_counts, values='Count', names='Video', 
                         title='Incidents by Video')
        st.plotly_chart(fig_video, use_container_width=True)
        
        # Raw data exploration
        st.subheader("Raw Data")
        if st.checkbox("Show Raw Data"):
            st.dataframe(df)
            
            # Export option
            if st.button("Export Data to CSV"):
                df.to_csv("anomaly_analytics_export.csv", index=False)
                st.success("Data exported to anomaly_analytics_export.csv")
    
    except Exception as e:
        st.error(f"Error generating analytics: {str(e)}")

def admin_panel():
    st.subheader("Admin Panel")
    
    tab1, tab2 = st.tabs(["User Management", "System Management"])
    
    with tab1:
        st.subheader("User Management")
        st.markdown("### Register New User")
        
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        role = st.selectbox("Role", ["user", "admin"])
        
        if st.button("Register User"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
            elif not new_username or not new_password:
                st.error("Username and password cannot be empty")
            else:
                if register_user(new_username, new_password, role):
                    st.success(f"User {new_username} registered successfully!")
                else:
                    st.error(f"Username '{new_username}' already exists")
                    
        # Display current users
        st.markdown("### Current Users")
        conn, cursor = init_user_db()
        cursor.execute("SELECT username, role FROM users")
        users = cursor.fetchall()
        conn.close()
        
        if users:
            users_df = pd.DataFrame(users, columns=["Username", "Role"])
            st.dataframe(users_df)
        else:
            st.info("No users found")
    
    with tab2:
        st.markdown("### System Management")
        
        # Clear all incidents
        if st.button("Clear All Incidents"):
            try:
                conn = sqlite3.connect('anomaly_analytics.db')
                cursor = conn.cursor()
                cursor.execute("DELETE FROM behavior_analytics")
                conn.commit()
                conn.close()
                st.success("All incidents have been cleared")
            except Exception as e:
                st.error(f"Error clearing database: {str(e)}")
        
        # Reset system
        st.warning("Danger Zone")
        if st.button("Reset Entire System"):
            try:
                # Clear databases
                conn = sqlite3.connect('anomaly_analytics.db')
                cursor = conn.cursor()
                cursor.execute("DELETE FROM behavior_analytics")
                conn.commit()
                conn.close()
                
                # Reset live data
                st.session_state['live_data'] = {
                    'timestamps': [],
                    'normal_count': [],
                    'anomaly_count': []
                }
                
                st.success("System has been reset")
                st.info("Note: User accounts were not deleted")
                
                # Remove image files
                import shutil
                image_dir = Path("anomaly_captures/incident_captures")
                if image_dir.exists():
                    shutil.rmtree(image_dir)
                    image_dir.mkdir(exist_ok=True)
            except Exception as e:
                st.error(f"Error resetting system: {str(e)}")

def main():
    if not st.session_state.get('logged_in', False):
        login_page()
    else:
        st.title(f"Anomaly Detection System")
        st.write(f"Welcome, {st.session_state['username']}!")
        
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['username'] = None
            st.rerun()  # Changed from experimental_rerun
            
        # Admin section
        if st.session_state.get('username') == 'admin':
            if st.checkbox("Show Admin Panel"):
                admin_panel()
                st.markdown("---")
        
        # Main tabs
        tabs = st.tabs(["Upload Video", "History", "Analytics Dashboard"])
        
        with tabs[0]:
            # Video upload section
            st.subheader("Upload Video for Analysis")
            uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
            
            if uploaded_file is not None:
                temp_path = "uploaded_video.mp4"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                detector = AnomalyDetector()
                detector.process_video(temp_path, uploaded_file.name)
        
        with tabs[1]:
            history_page()
            
        with tabs[2]:
            analytics_dashboard()

if __name__ == "__main__":
    main()