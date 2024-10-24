import streamlit as st
import time

# Function that simulates an independent loop
def independent_loop():
    for i in range(10):  # Change the range for longer execution
        st.session_state.loop_progress = i + 1  # Update session state with progress
        time.sleep(1)  # Simulate a time-consuming task
    st.session_state.loop_running = False  # Mark the loop as finished

# Streamlit app
st.title("Independent Loop Example")

# Initialize session state
if 'loop_running' not in st.session_state:
    st.session_state.loop_running = False
if 'loop_progress' not in st.session_state:
    st.session_state.loop_progress = 0

# Start button
if st.button("Start Loop"):
    if not st.session_state.loop_running:
        st.session_state.loop_running = True
        st.session_state.loop_progress = 0
        st.experimental_rerun()  # Rerun to display the progress

# Run the independent loop if it is running
if st.session_state.loop_running:
    independent_loop()

# Display progress
if st.session_state.loop_running:
    st.write(f"Loop is running... Progress: {st.session_state.loop_progress}/10")
else:
    st.write("Loop finished or not started yet.")
