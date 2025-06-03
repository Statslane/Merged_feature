import streamlit as st
import cv2
import easyocr
import numpy as np
import pandas as pd
import tempfile
import os
from ultralytics import YOLO
from norfair import Detection, Tracker
import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load different models for each page
video_analyzer_model = YOLO('Models/best_possession_model.pt')  # Model for Video Analyzer page
shot_analysis_model = YOLO('Models/best_scoring_model.pt')    # Model for Shot Analysis page
advance_video_analysis = YOLO('Models/bestjersey.pt')
# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Initialize NorFair Tracker
def euclidean(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

tracker = Tracker(distance_function=euclidean, distance_threshold=50)

# Define colors for each class
colors = {
    0: (0, 165, 255),   # Ball - Orange
    1: (0, 0, 255),     # Hoop - Red
    2: (128, 0, 128),   # Jersey - Purple
    3: (0, 255, 255),   # Referee - Cyan
    4: (255, 255, 0),   # Team 2 with Ball - Yellow
    5: (255, 0, 0),     # Team 1 - Blue
    6: (0, 255, 0),     # Team 1 with Ball - Green
    7: (255, 165, 0)    # Team 2 - Orange
}

# Function to extract text (jersey number) using EasyOCR
def extract_jersey_number(jersey_crop):
    gray = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    results = reader.readtext(gray)

    for (bbox, text, prob) in results:
        if prob > 0.35:
            text = ''.join(filter(str.isdigit, text))
            if text:
                return text
    return None

# Streamlit Page Config
st.set_page_config(page_title="Basketball Video TagAI", layout="wide")

# Sidebar navigation
st.markdown("""
    <style>
    /* Target the sidebar's radio button text */
    [data-testid="stSidebar"] [data-baseweb="radio"] > div {
        font-size: 40px !important;
    }

    /* Optional: change the radio header/title */
    [data-testid="stSidebar"] .stRadio label {
        font-size: 42px !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar radio input
page = st.sidebar.radio(
    "Select a Page",
    ["Home", "Video Analyzer", "Shot Analysis", "Advance Analysis", "Player Stats", "Team Stats", "Opponent Scouting", "Training & Development"]    
)
# Home Page
if page == "Home":
    st.image("logos/logo.png", width=150)
    st.title("Basketball Video TagAI")
    st.markdown("""
    Welcome to the **Basketball Video TagAI**! 🏀🎥

    This tool allows you to:
    - Upload and analyze basketball game videos
    - Detect ball possession for each team
    - Extract jersey numbers using OCR
    - Track player movement with real-time object detection and tracking

    Get started by selecting the **Video Analyzer** or **Shot Analysis** page from the sidebar.
    """)

# Video Analyzer Page
elif page == "Video Analyzer":
    st.title("Basketball Video Analyzer")
    st.write("Upload a video to detect ball possession and count team possession times.")
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save uploaded file to temp path
        temp_input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(temp_input_path)

        cap = cv2.VideoCapture(temp_input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        jersey_numbers = {}
        team1_with_ball_count = 0
        team2_with_ball_count = 0
        previously_counted = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = video_analyzer_model(frame)  # Use video_analyzer_model here

            detections = []
            jersey_bboxes = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls == 2:
                        jersey_crop = frame[y1:y2, x1:x2]
                        jersey_number = extract_jersey_number(jersey_crop)
                        jersey_bboxes.append((x1, y1, x2, y2, jersey_number))

                    detections.append(Detection(points=np.array([(x1 + x2) / 2, (y1 + y2) / 2]), data={"cls": cls, "bbox": (x1, y1, x2, y2)}))

            tracked_objects = tracker.update(detections)

            for track in tracked_objects:
                center_point = track.estimate
                x, y = center_point[0]
                data = track.last_detection.data
                x1, y1, x2, y2 = data["bbox"]
                cls = data["cls"]
                track_id = track.id

                if cls in [4, 6]:  # Team with ball
                    if track_id not in previously_counted:
                        if cls == 4:
                            team2_with_ball_count += 1
                        elif cls == 6:
                            team1_with_ball_count += 1
                        previously_counted.add(track_id)

                if cls in [4, 5, 6, 7]:
                    jersey_number = None
                    for (jx1, jy1, jx2, jy2, jnum) in jersey_bboxes:
                        if x1 < jx1 < x2 and y1 < jy1 < y2 and jnum is not None:
                            jersey_number = jnum
                            break

                    if track_id in jersey_numbers:
                        prev_number = jersey_numbers[track_id]
                        if jersey_number is not None:
                            jersey_numbers[track_id] = jersey_number
                        else:
                            jersey_number = prev_number
                    else:
                        jersey_numbers[track_id] = jersey_number

                    final_number = jersey_numbers.get(track_id, " ")
                    color = colors.get(cls, (255, 255, 255))
                    label = f"{video_analyzer_model.names[cls]} - {final_number}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    color = colors.get(cls, (255, 255, 255))
                    label = f"{video_analyzer_model.names[cls]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)

        cap.release()
        out.release()

        # Display possession in columns
        total_possession = team1_with_ball_count + team2_with_ball_count
        if total_possession == 0:
            team1_percent = 0
            team2_percent = 0
        else:
            team1_percent = (team1_with_ball_count / total_possession) * 100
            team2_percent = (team2_with_ball_count / total_possession) * 100

        col1, col2 = st.columns(2)
        col1.metric(label="Team 1 Possession %", value=f"{team1_percent:.2f}%")
        col2.metric(label="Team 2 Possession %", value=f"{team2_percent:.2f}%")

        # Provide download button
        with open(temp_output_path, "rb") as f:
            st.download_button("📥 Download Processed Video", f, file_name="processed_video.mp4", mime="video/mp4")

        # Clean up temp files
        os.remove(temp_input_path)
        os.remove(temp_output_path)

# Shot Analysis Page
elif page == "Shot Analysis":
    st.title("🏀 Basketball Shot Analysis")
    st.write("Upload a video to analyze basketball shots (attempts, makes, misses).")
    show_preds = st.sidebar.checkbox("Show Predictions on Video", value=True)


    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        with open(temp_input, "wb") as f:
            f.write(uploaded_file.read())

        st.video(temp_input)

        with st.spinner("🔍 Processing Video..."):
            cap = cv2.VideoCapture(temp_input)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

            attempt_count = 0
            make_count = 0
            miss_count = 0
            last_event = "none"

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = shot_analysis_model(frame, verbose=False)
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        if cls == 0:  # Attempt
                            label = f"Attempt {conf:.2f}"
                            color = (255, 255, 0)
                            if last_event != "attempt":
                                attempt_count += 1
                                last_event = "attempt"

                        elif cls == 2:  # Make
                            label = f"Make {conf:.2f}"
                            color = (0, 255, 0)
                            if last_event == "attempt":
                                make_count += 1
                                last_event = "make"

                        elif cls == 3:  # Miss
                            label = f"Miss {conf:.2f}"
                            color = (255, 0, 0)
                            if last_event == "attempt":
                                miss_count += 1
                                last_event = "miss"

                        elif cls == 1:  # Hoop
                            label = f"Hoop {conf:.2f}"
                            color = (255, 255, 255)

                        else:
                            continue

                        if show_preds:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                out.write(frame)

            cap.release()
            out.release()

        st.success("✅ Video Processed!")

        # Results
        st.subheader("📊 Detection Summary")
        st.metric("Attempts", attempt_count)
        st.metric("Baskets", make_count)
        st.metric("Misses", miss_count)
        st.metric("Accuracy", (make_count / (make_count + miss_count)) * 100)

        # Plot
        fig, ax = plt.subplots()
        ax.bar(["Attempts", "Baskets", "Misses"], [attempt_count, make_count, miss_count], color=["gold", "green", "red"])
        ax.set_ylabel("Count")
        ax.set_title("Basketball Shot Analysis")
        st.pyplot(fig)

        # Download
        with open(temp_output, "rb") as f:
            st.download_button("📥 Download Processed Video", f, file_name="processed_video.mp4", mime="video/mp4")

        os.remove(temp_input)
        os.remove(temp_output)

elif page == 'Advance Analysis':

    # Configuration
    EXTRA_FRAMES = 5  # Number of extra frames to add to each clip
    
    # Initialize session state for caching
    if 'clips' not in st.session_state:
        st.session_state.clips = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'frame_numbers' not in st.session_state:
        st.session_state.frame_numbers = {'make': [], 'miss': []}
    
        # Initialize DeepSort for tracking only players (class 5)
        deep_sort = DeepSort(
            max_age=10,
            n_init=3,
            max_cosine_distance=0.3,
            nn_budget=100
        )
    # Function to create video clips
    def create_video_clips(input_video, frame_events, output_dir):
        cap = cv2.VideoCapture(input_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        clips = []
        make_count = 1  # Counter for make clips
        miss_count = 1  # Counter for miss clips

        # Sort all frame numbers and their types
        all_events = [(frame, 'make') for frame in frame_events['make']] + [(frame, 'miss') for frame in frame_events['miss']]
        all_events.sort()  # Sort by frame number

        # Add start frame (1) if not already included
        if not all_events or all_events[0][0] != 1:
            all_events.insert(0, (1, 'start'))

        # Create clips
        for i in range(len(all_events)):
            start_frame = all_events[i][0]
            event_type = all_events[i][1]
            end_frame = all_events[i + 1][0] if i + 1 < len(all_events) else total_frames
            # Extend end_frame by EXTRA_FRAMES, but not beyond total frames
            extended_end_frame = min(end_frame + EXTRA_FRAMES, total_frames)

            # Determine clip name based on event type
            if event_type == 'make':
                clip_name = f"Make Clip {make_count}.mp4"
                make_count += 1
            elif event_type == 'miss':
                clip_name = f"Miss Clip {miss_count}.mp4"
                miss_count += 1
            else:
                clip_name = f"Start Clip.mp4"  # For the first clip if it starts at frame 1

            clip_path = os.path.join(output_dir, clip_name)

            # Reset video capture to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
            out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            # Write frames to clip
            for frame_num in range(start_frame, extended_end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            out.release()
            clips.append((clip_name, clip_path))

        cap.release()
        return clips

    # Page setup
    # st.set_page_config(page_title="🏀 Basketball Analyzer", layout="wide")
    st.image("logos/logo.png", width=150)
    st.title("Basketball Video TagAI")

    # Sidebar
    st.sidebar.title("🧠 AI Video Analyzer")
    st.sidebar.markdown("This tool detects:")
    st.sidebar.markdown("- 🟡 Ball Attempt")
    st.sidebar.markdown("- ⚪ Hoop Position")
    st.sidebar.markdown("- 🟢 Successful Basket")
    st.sidebar.markdown("- 🔴 Missed Attempt")
    st.sidebar.markdown("- 🟠 Team A Detection")
    st.sidebar.markdown("- 🔵 Team B Detection")
    st.sidebar.markdown("- 📏 Player Tracking with ID")
    show_preds = st.sidebar.checkbox("Show Predictions on Video", value=True)

# File uploader
uploaded_file = st.file_uploader("Upload a Basketball Video", type=["mp4", "avi", "mov"])

# Process video only if not already processed
if uploaded_file and st.session_state.clips is None:
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    temp_dir = tempfile.mkdtemp()  # Directory for clips

    with open(temp_input, "wb") as f:
        f.write(uploaded_file.read())

    st.video(temp_input)

    with st.spinner("🔍 Processing Video..."):
        cap = cv2.VideoCapture(temp_input)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        slow_motion_factor = 3
        out = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*"mp4v"), fps // slow_motion_factor, (width, height))

        # Counters and frame lists
        teamA_make = teamB_make = teamA_miss = teamB_miss = 0
        stable_total_make = stable_total_miss = 0
        make_frames = []  # List to store frame numbers of makes
        miss_frames = []  # List to store frame numbers of misses
        stable_event_running = False
        stable_current_event = None
        last_team_detected = None
        frame_count = 0  # Track current frame number
        last_event_frame = 0  # Track the last frame of the current event sequence

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            for _ in range(slow_motion_factor - 1):
                cap.read()
                frame_count += 1

            results = advance_video_analysis(frame, verbose=False)
            detections_for_deepsort = []
            make_in_frame = miss_in_frame = teamA_in_frame = teamB_in_frame = False

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if cls == 0:  # Ball
                        label = f"Ball {conf:.2f}"
                        color = (255, 255, 0)

                    elif cls == 2:  # Make
                        label = f"Make {conf:.2f}"
                        color = (0, 255, 0)
                        make_in_frame = True

                    elif cls == 3:  # Miss
                        label = f"Miss {conf:.2f}"
                        color = (255, 0, 0)
                        miss_in_frame = True

                    elif cls == 1:  # Hoop
                        label = f"Hoop {conf:.2f}"
                        color = (255, 255, 255)

                    elif cls == 6:  # TeamA
                        label = f"TeamA {conf:.2f}"
                        color = (255, 140, 0)
                        teamA_in_frame = True

                    elif cls == 7:  # TeamB
                        label = f"TeamB {conf:.2f}"
                        color = (0, 191, 255)
                        teamB_in_frame = True

                    elif cls == 5:  # Player
                        if conf > 0.4:
                            detections_for_deepsort.append(([x1, y1, x2 - x1, y2 - y1], conf, None))
                        continue

                    else:
                        continue

                    if show_preds:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Apply DeepSORT to player boxes only
            tracks = deep_sort.update_tracks(detections_for_deepsort, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f"Player {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Team tracking
            if teamA_in_frame:
                last_team_detected = "teamA"
            if teamB_in_frame:
                last_team_detected = "teamB"

            # Event tracking
            if make_in_frame and not stable_event_running:
                stable_current_event = "make"
                stable_event_running = True
                last_event_frame = frame_count  # Start of make sequence

            elif miss_in_frame and not stable_event_running:
                stable_current_event = "miss"
                stable_event_running = True
                last_event_frame = frame_count  # Start of miss sequence

            elif (make_in_frame or miss_in_frame) and stable_event_running:
                # Update last_event_frame if the same event continues
                if (make_in_frame and stable_current_event == "make") or (miss_in_frame and stable_current_event == "miss"):
                    last_event_frame = frame_count

            elif not make_in_frame and not miss_in_frame and stable_event_running:
                # Event sequence has ended, save the last frame
                if stable_current_event == "make":
                    make_frames.append(last_event_frame)  # Save last frame of make sequence
                    stable_total_make += 1
                    if last_team_detected == "teamA":
                        teamA_make += 1
                    elif last_team_detected == "teamB":
                        teamB_make += 1
                elif stable_current_event == "miss":
                    miss_frames.append(last_event_frame)  # Save last frame of miss sequence
                    stable_total_miss += 1
                    if last_team_detected == "teamA":
                        teamA_miss += 1
                    elif last_team_detected == "teamB":
                        teamB_miss += 1
                stable_event_running = False
                stable_current_event = None

            out.write(frame)
            frame_count += 1

        # Handle case where video ends during an event
        if stable_event_running:
            if stable_current_event == "make":
                make_frames.append(last_event_frame)
                stable_total_make += 1
                if last_team_detected == "teamA":
                    teamA_make += 1
                elif last_team_detected == "teamB":
                    teamB_make += 1
            elif stable_current_event == "miss":
                miss_frames.append(last_event_frame)
                stable_total_miss += 1
                if last_team_detected == "teamA":
                    teamA_miss += 1
                elif last_team_detected == "teamB":
                    teamB_miss += 1

        cap.release()
        out.release()

        # Create video clips
        frame_events = {'make': make_frames, 'miss': miss_frames}
        clips = create_video_clips(temp_input, frame_events, temp_dir)

        # Cache results in session state
        st.session_state.clips = clips
        st.session_state.processed_data = {
            'teamA_make': teamA_make,
            'teamB_make': teamB_make,
            'teamA_miss': teamA_miss,
            'teamB_miss': teamB_miss,
            'stable_total_make': stable_total_make,
            'stable_total_miss': stable_total_miss
        }
        st.session_state.frame_numbers = frame_events
        st.session_state.temp_output = temp_output
        st.session_state.temp_dir = temp_dir
        st.session_state.temp_input = temp_input

    st.success("✅ Video Processed!")

# Display results if available
if st.session_state.processed_data:
    # Results
    st.subheader("📊 Detection Summary")
    data = {
        "Category": [
            "Total Attempts",
            "Total Makes",
            "Total Misses",
            "Team A Makes",
            "Team A Misses",
            "Team B Makes",
            "Team B Misses",
        ],
        "Count": [
            st.session_state.processed_data['stable_total_make'] + st.session_state.processed_data['stable_total_miss'],
            st.session_state.processed_data['stable_total_make'],
            st.session_state.processed_data['stable_total_miss'],
            st.session_state.processed_data['teamA_make'],
            st.session_state.processed_data['teamA_miss'],
            st.session_state.processed_data['teamB_make'],
            st.session_state.processed_data['teamB_miss'],
        ]
    }
    df = pd.DataFrame(data)
    st.table(df)

    # # Display frame numbers
    # st.subheader("🖼️ Frame Numbers")
    # st.write("**Make Frames:**", ", ".join(map(str, st.session_state.frame_numbers['make'])) if st.session_state.frame_numbers['make'] else "No makes detected")
    # st.write("**Miss Frames:**", ", ".join(map(str, st.session_state.frame_numbers['miss'])) if st.session_state.frame_numbers['miss'] else "No misses detected")

    fig, ax = plt.subplots()
    ax.bar(data["Category"], data["Count"], color=["gold", "green", "red", "orange", "salmon", "skyblue", "deepskyblue"])
    ax.set_ylabel("Count")
    ax.set_title("Basketball Shot Analysis")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown("### Download Processed Video")
    with open(st.session_state.temp_output, "rb") as f:
        st.download_button("📥 Download Processed Video", f, file_name="processed_video.mp4", mime="video/mp4", key="processed_video")
        
    # Display clips in a grid
    st.subheader("🎥 Generated Clips")
    col1, col2 = st.columns(2)  # Two columns for Make and Miss clips

    with col1:
        st.markdown("### Make Clips")
        for clip_name, clip_path in st.session_state.clips:
            if "Make Clip" in clip_name:
                
                with open(clip_path, "rb") as f:
                    st.download_button(f"📥 Download {clip_name}", f, file_name=clip_name, mime="video/mp4", key=clip_name)


    with col2:
        st.markdown("### Miss Clips")
        for clip_name, clip_path in st.session_state.clips:
            if "Miss Clip" in clip_name:
                
                with open(clip_path, "rb") as f:
                    st.download_button(f"📥 Download {clip_name}", f, file_name=clip_name, mime="video/mp4", key=clip_name)

    # Handle Start Clip separately
    # for clip_name, clip_path in st.session_state.clips:
    #     if "Start Clip" in clip_name:
    #         st.markdown("### Start Clip")
            
    #         with open(clip_path, "rb") as f:
    #             st.download_button(f"📥 Download {clip_name}", f, file_name=clip_name, mime="video/mp4", key=clip_name)


    # Cleanup
    os.remove(st.session_state.temp_input)
    os.remove(st.session_state.temp_output)
    for _, clip_path in st.session_state.clips:
        os.remove(clip_path)
    os.rmdir(st.session_state.temp_dir)
    # Clear session state after cleanup
    st.session_state.clips = None
    st.session_state.processed_data = None
    st.session_state.frame_numbers = {'make': [], 'miss': []}

