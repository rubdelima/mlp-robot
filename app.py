import streamlit as st
import serial.tools.list_ports
import cv2
import numpy as np
from utils.robot import Robot

def initialize_robot(port):
    if port:
        try:
            return Robot(port)
        except Exception as e:
            st.error(f"Erro ao conectar com o braço: {e}")
            return None
    return None

def safe_close_robot(robot):
    if robot is not None:
        try:
            robot.close()
            return True
        except Exception as e:
            st.error(f"Erro ao fechar conexão com o braço: {e}")
            return False
    return True

# Configuração de estado da sessão para o robô
if 'robot' not in st.session_state:
    st.session_state.robot = None
if 'robot_port' not in st.session_state:
    st.session_state.robot_port = None
if 'frame_placeholder' not in st.session_state:
    st.session_state.frame_placeholder = None

# Interface do Streamlit
st.set_page_config(layout="wide")

# Divisão em 2 colunas
col1, col2 = st.columns([1, 1])

with col1:
    pass

with col2:
    st.header("Controle")
    
    ports = [port.device for port in serial.tools.list_ports.comports()]
    
    selected_port = st.selectbox("Selecione a Porta COM", ports)
    
    connect_col, disconnect_col, reset_col = st.columns([2,2,1])
    
    with connect_col:
        if st.button("Conectar", use_container_width=True):
            if selected_port:
                if st.session_state.robot is not None:
                    safe_close_robot(st.session_state.robot)
                
                st.session_state.robot = initialize_robot(selected_port)
                
                if st.session_state.robot is not None:
                    st.session_state.robot_port = selected_port
                    st.success(f"Conectado em: {selected_port}")

            else:
                st.info("Selecione uma porta COM")
    
    with disconnect_col:
        if st.button("Desconectar", use_container_width=True):
            if st.session_state.robot is not None:
                if safe_close_robot(st.session_state.robot):
                    st.session_state.robot = None
                    st.session_state.robot_port = None
                    st.success("Desconectado")

    with reset_col:
        if st.button("Reset", use_container_width=True, disabled=st.session_state.robot is None):
            if st.session_state.robot is not None:
                st.session_state.robot.reset()
                st.success("Reset OK")
    
    if st.session_state.robot is not None:
        st.success(f"Braço conectado na porta {st.session_state.robot_port}")
    else:
        st.warning("Braço não conectado")
    
    # Exibição das métricas dos eixos do robô
    metrics_cols = st.columns(4)
    
    if st.session_state.robot is not None:
        robot = st.session_state.robot
        for i in range(4):
            metrics_cols[i].metric(f"Eixo {i}", str(getattr(robot, f'axis{i}')))
    else:
        for i in range(4):
            metrics_cols[i].metric(f"Eixo {i}", "--")
    
    submit_button = False
    
    if st.session_state.robot is not None:
        with st.form("robot_form"):
            robot = st.session_state.robot
            
            axis0 = st.slider("Eixo 0", 0, 179, robot.axis0)
            axis1 = st.slider("Eixo 1", 0, 179, robot.axis1)
            axis2 = st.slider("Eixo 2", 0, 179, robot.axis2)
            axis3 = st.slider("Eixo 3", 0, 179, robot.axis3)

            submit_button = st.form_submit_button("Atualizar Posição")
    
    if submit_button and st.session_state.robot is not None:
        try:
            st.session_state.robot.move_to(axis0, axis1, axis2, axis3)
            st.success("Comando enviado!")
            
            for i in range(4):
                metrics_cols[i].metric(f"Eixo {i}", str(getattr(st.session_state.robot, f'axis{i}')))
        
        except Exception as e:
            st.error(f"Erro ao enviar comando: {e}")


# Coluna de visualização
with col1:
    st.header("Visualização")
    
    webcam_index = st.selectbox("Selecione a webcam", [None, 0, 1])

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()

    
    # Placeholder para ArUco 0
    ar0_placeholder = st.empty()
    ar0_placeholder.metric("Posição ArUco 0", "(--, --)")

    # Placeholder para ArUco 1
    
    ar1_placeholders : list[str] = []
    ar1_placeholder = st.empty()

    # Placeholder para a imagem
    placeholder = st.empty()
    
    freeze_button = st.button("Freeze")

    frozen = False
    
    if webcam_index is not None:
        cap = cv2.VideoCapture(webcam_index)

        if not cap.isOpened():
            st.error("Não foi possível acessar a câmera.")

        while True:
            if not frozen:
                ret, frame = cap.read()
                if not ret:
                    break

                corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

                position_aruco_0 = ""
                position_aruco_1 = []

                if len(corners) > 0:
                    frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                    for i, id in enumerate(ids):
                        if id == 0:
                            x, y = np.mean(corners[i][0], axis=0).astype(int)
                            position_aruco_0 = f"({x},{y})"
                        elif id == 1:
                            x, y = np.mean(corners[i][0], axis=0).astype(int)
                            position_aruco_1.append(f"({x},{y})")

                ar0_placeholder.metric("Posição ArUco 0", position_aruco_0 if position_aruco_0 else "(--, --)")

                if position_aruco_1:
                    ar1_cols = ar1_placeholder.columns(len(position_aruco_1))
                    for i, (ar1_pos, col) in enumerate(zip(position_aruco_1, ar1_cols)):
                        col.metric(f"Posição ArUco 1 ({i + 1})", ar1_pos)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                placeholder.image(frame, use_container_width=True)

            if freeze_button:
                frozen = not frozen

        cap.release()
