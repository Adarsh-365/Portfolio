# Import the necessary libraries
import streamlit as st

# Set the page configuration to adjust the layout width
st.set_page_config(
    page_title="Simple Streamlit App",
    layout="wide",  # Use 'wide' to increase the app's width
    
)



st.sidebar.image: st.sidebar.image("image_circle.png", use_container_width=True) 

st.sidebar.markdown: st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <h2>Adarsh Vishnu Tayde</h2>
        <h2>Mtech. CSIS (NIT Warangal)</h2>
        </div>
    <div style="text-align: left;">
        <p>📞 +91-7378693514</p>
<p>✉️ adarshtayde9011@gmail.com</p>

<p><i class="fa-brands fa-github"></i>   https://github.com/Adarsh-365</p>
<p><i class="fa-brands fa-linkedin"></i>   https://www.linkedin.com/in/adarsh-tayde-nitw-csis</p>
</div>
    """,
    unsafe_allow_html=True
)



st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """,
    unsafe_allow_html=True
)

    
   

st.header("Education" , divider=True)



cols = st.columns(2,gap="Large")
cols = st.columns([8, 2])  # 80% and 20%
cols[0].markdown("### • National Institute of Technology, Warangal (Telangana)")
cols[0].markdown("> M.Tech. Computer Science and Information Technology")
cols[1].markdown("### 2024-2026")
cols[1].markdown("#### ")

cols = st.columns(2,gap="Large")
cols = st.columns([8, 2])  # 80% and 20%
cols[0].markdown("### • Anuradha Engineering College, Chikhli (Maharashtra)")
cols[0].markdown("> B.E. Information Technology")
cols[1].markdown("### 2022")
cols[1].markdown("#### 9.33 CGPA")


cols = st.columns(2,gap="Large")
cols = st.columns([8, 2])  # 80% and 20%
cols[0].markdown("### • Shri Shivaji Science and Arts College, Chikhli")
cols[0].markdown("> Higher Secondary Education, Maharashtra")
cols[1].markdown("### 2018")
cols[1].markdown("#### 83.85 %")

cols = st.columns(2,gap="Large")
cols = st.columns([8, 2])  # 80% and 20%
cols[0].markdown("### • Shri Shivaji High School Chikhli")
cols[0].markdown("> Board of Secondary Education, Maharashtra")
cols[1].markdown("### 2016")
cols[1].markdown("#### 84.80 %")



st.header("Experience" , divider=True)
cols = st.columns(2,gap="Large")
cols = st.columns([8, 2])  # 80% and 20%
cols[0].markdown("### • Triangle Simulation Pvt. Ltd.")
cols[0].markdown("> Software Engineer")
cols[1].markdown("#### Aug-2022 to July-2024")
cols[1].markdown("#### Mumbai (Wadala)")

st.markdown("""
            - Work with :blue[Central University of Jammu] to Developed Hydro-Power Plant simulator Tetbed.
- Work with :blue[IIT Karagpur] to Developed Level Control System and Flow Control System Simulator.
- Work with :blue[VJTI Mumbai] for POC of Power Grid Simulator Testbed.

            """)


st.header("Projects" , divider=True)

cols = st.columns(2,gap="Large")
cols = st.columns([8, 2])  # 80% and 20%
cols[0].markdown("### • Hydro-Power Plant Simulator Testbed")
cols[0].markdown("> Central University of Jammu")
cols[1].markdown("#### Dec-2022 to May-2023")
cols[1].markdown("#### Jammu ")


st.markdown("""
            - Developed a Hydro Power Plant Simulator integrating PLCs :blue[(Siemens and Allen Bradley)], HMIs, RTUs, SCADA
systems, and servers.
- Designed a user-friendly GUI to run Simulator used :blue[**Modbus Protocol**] for communication.
- Programmed :blue[**Siemens PLC**] with :blue[**ladder logic**]; integrated PLCs with :blue[**SCADA**] and :blue[**HMI**] systems.
- Created :blue[**HMI**] and **SCADA** mimics for data visualization.
- Trained Junior and Senior Research Fellows on system operations and testing.

            """)



cols = st.columns(2,gap="Large")
cols = st.columns([8, 2])  # 80% and 20%
cols[0].markdown("### • Level Control System and Flow Control System Simulator")
cols[0].markdown("> IIT Karagpur")
cols[1].markdown("#### March-2024")
cols[1].markdown("#### Karagpur (West Bengal) ")


st.markdown("""
            - Developed :blue[**Labview**] based Simulator to study labscale systems like Level Control Syteam and Flow Control System.
- To study we used :blue[**PID Control Card**] of PPI with Modbus Serial protocol.


            """)


cols = st.columns(2,gap="Large")
cols = st.columns([8, 2])  # 80% and 20%
cols[0].markdown("### • POC of Grid Simulator Testbed")
cols[0].markdown("> VJTI Mumbai")
cols[1].markdown("#### Oct-2023 to July-2024")
cols[1].markdown("#### Mumbai")


st.markdown("""
            - Developed a Demo of Power Grid Simulator integrating hardware Like  Relay(IED’s), Doble, and PMU.
- Designed a user-friendly GUI to run Load Flow Analysis and Transient Stability Simulation.
- To communicate with hardware like Relay used :blue[**IEC61850**] protocol Modbus, for PMU used :blue[**c37.118**],for testing with
Arbitrary Waveform Generator used :blue[**SCPI**],for micro-controller :blue[**VISA**].
            """)



st.header("Technical Skills" , divider=True)

cols = st.columns(2)
cols = st.columns([5, 5])  # 80% and 20%
cols[0].markdown("### **Languages** : ")
cols[0].markdown("### **Developer Tools** : ")
cols[0].markdown("### **Python Libraries** : ")

cols[1].markdown("#### Python, Cpp, Labview")
cols[1].markdown("#### Git, Github, Spyder, VS Code ")
cols[1].markdown("#### Numpy,pandas,matplotlib,pyqt5,tkinter,streamlit etc")


st.header("Positions of Responsibility" , divider=True)
cols = st.columns(2,gap="Large")
cols = st.columns([8, 2])  # 80% and 20%
cols[0].markdown("#### • Position, Head of the Chess Region Chikhli ")
cols[1].markdown("#### 2019-2022")

st.header("Achievements" , divider=True)


cols = st.columns(2,gap="Large")
cols = st.columns([8, 2])  # 80% and 20%
cols[0].markdown("#### • Achievement :red[Gold Medalist] in Bachelor of Engineering (B.E.) in Course of Information Technology ")
cols[1].markdown("#### 2022")
cols[0].markdown("#### • Achievement Got :red[Second Prize] in inter-branch chess ")
cols[1].markdown("#### 2019")
cols[0].markdown("#### • Achievement Got Opportunity to help as :red[Arbiter in District Level Chess from District Sports Council] ")
cols[1].markdown("#### 2019")
