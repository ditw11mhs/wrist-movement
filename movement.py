import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.express as px
from pandas import DataFrame as df

class Main:
    def main(self):
        
        st.markdown("# 3D Motion")
        
        time = np.arange(0, 5, 0.001)
        f_shoulder = 0.25
        f_elbow = 0.25
        angle_x_shoulder = np.pi/2 * np.abs(np.sin(2 * np.pi * f_shoulder * time))
        angle_y_shoulder = angle_x_shoulder / 2
        angle_x_elbow = np.pi * np.abs(np.sin(2 * np.pi * f_elbow * time))
        angle_y_lower_arm = angle_x_elbow/ 2

        
        self.one()
        self.two()
        self.three()

        

    def webgl_component(self, camera_radius, camera_theta, camera_phi, H, M):
        l1 = 0.172*H
        l2 = 0.157*H
        components.html(
            f"""
            <canvas id="canvas" width = "640" height = "640"></canvas>
            <script src="https://twgljs.org/dist/3.x/twgl-full.min.js"></script>
            <script>
            const m4 = twgl.m4;
            const v3 = twgl.v3;
            const gl = document.querySelector("canvas").getContext("webgl");
            const vs = `
            attribute vec4 position;
            attribute vec3 normal;
            uniform mat4 u_projection;
            uniform mat4 u_view;
            uniform mat4 u_model;
            varying vec3 v_normal;
            void main() {{
            gl_Position = u_projection * u_view * u_model * position;
            v_normal = mat3(u_model) * normal; // better to use inverse-transpose-model
            }}
            `
            const fs = `
            precision mediump float;
            varying vec3 v_normal;
            uniform vec3 u_lightDir;
            uniform vec3 u_color;
            void main() {{
            float light = dot(normalize(v_normal), u_lightDir) * .5 + .5;
            gl_FragColor = vec4(u_color * light, 1);
            }}
            `;
            // compiles shaders, links program, looks up attributes
            const programInfo = twgl.createProgramInfo(gl, [vs, fs]);
            // calls gl.createBuffer, gl.bindBuffer, gl.bufferData
            sphereRad =3;
            const cubeBufferInfo = twgl.primitives.createCubeBufferInfo(gl, 1);
            const sphereBufferInfo = twgl.primitives.createSphereBufferInfo(gl,sphereRad,200,200);
            const cylinderBufferInfo = twgl.primitives.createCylinderBufferInfo(gl,sphereRad/2,{l1}+2*sphereRad,200,200);
            const cylinder2BufferInfo = twgl.primitives.createCylinderBufferInfo(gl,sphereRad/10,(2*sphereRad),200,200);
            const truncatCylinderBufferInfo = twgl.primitives.createTruncatedConeBufferInfo(gl,sphereRad/3,sphereRad/2,{l2}+2*sphereRad,200,200);
            const cylinder3BufferInfo = twgl.primitives.createCylinderBufferInfo(gl,sphereRad/2,(2*sphereRad),200,200);
           
            r = {camera_radius};
            theta_camera = {camera_theta}*Math.PI/180;
            phi = {camera_phi}*Math.PI/180;
           
            x = r*Math.sin(theta_camera)*Math.cos(phi);
            z = r*Math.sin(theta_camera)*Math.sin(phi);
            y = -r*Math.cos(theta_camera);
            
            const stack = [];
            const color = [1, 1,1];
            const lightDir = v3.normalize([x, y, z]);
           
           
            
            function render(time) {{
            
            time *= 0.001;
            f_shoulder = 0.25;
            f_elbow = 0.25;
            
            rotate_shoulder_x = 0;
            rotate_shoulder_y = 0;
            rotate_elbow_x = 0;
            rotate_lower_arm = Math.sin(time);
            
            twgl.resizeCanvasToDisplaySize(gl.canvas);
            gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
            gl.enable(gl.DEPTH_TEST);
            gl.enable(gl.CULL_FACE);
            gl.useProgram(programInfo.program);
            const fov = Math.PI * .25;
            const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
            const zNear = 0.01;
            const zFar = 1000;
            const projection = m4.perspective(fov, aspect, zNear, zFar);
            const cameraPosition = [x, y, z];
            
            const target = [0, 0, -({l1}/2+sphereRad+{l1/2}+sphereRad)];
            const up = [0, 1, 0];
            const camera = m4.lookAt(cameraPosition, target, up);
            const view = m4.inverse(camera);
            
            // make base position for shoulder
            let m = m4.translation([0, 0, 0]);
            pushMatrix(m);
            {{
                //Shoulder Sphere
                m = m4.rotateX(m, rotate_shoulder_x);
                m = m4.rotateY(m, rotate_shoulder_y);
                drawSphere(projection, view, m);
                pushMatrix(m);
                {{
                    // Upper Arm
                    m = m4.translate(m, [0, -({l1}/2+sphereRad), 0]);
                    drawCylinder(projection, view, m);
                    pushMatrix(m);
                    {{
                    // Elbow
                    m = m4.translate(m, [0, -({l1/2}+sphereRad), 0]);
                    m = m4.rotateX(m, rotate_shoulder_y);
                    drawSphere(projection, view, m);
                    pushMatrix(m);
                    {{
                        
                        // Lower Arm
                        m = m4.translate(m, [0, -({l2/2}+sphereRad), 0]);
                        m = m4.rotateY(m, rotate_lower_arm);
                        drawCone(projection, view, m);
                        
                        pushMatrix(m);
                        {{
                            //Hand and thumb
                            m = m4.translate(m, [0, -({l2/2}+sphereRad), 0]);
                            m = m4.rotateX(m, rotate_lower_arm);
                            m= m4.rotateZ(m,rotate_lower_arm)
                            drawSphere(projection, view, m);
                            
                            m = m4.translate(m, [0, -sphereRad, 0]);
                            drawCylinder3(projection, view, m);
                            
                           
                            m = m4.rotateZ(m, 90);
                             m = m4.translate(m,[-3,-3,0]);
                            drawCylinder2(projection, view, m);
                            pushMatrix(m);
                        }}
                    }}
                    }}
                }}
            }}
            m = popMatrix();
            requestAnimationFrame(render);
            }}
            requestAnimationFrame(render);
            
            
            function pushMatrix(m) {{
            stack.push(m);
            }}
            
            
            function popMatrix() {{
            return stack.pop();
            }}
            
            //function MotionEquation(thetaa,thetadota,phia,phidota){{
            //    thetadotdot=((Fmus+torque/180*pi)+m*sqr(length)/8*phidot*phidot*sin(theta)*cos(theta)-m*gravity*length/2*sin(theta))/(m*sqr(length)/4+inertia);
            //    phidotdot=(torque1-m*sqr(length)/4*phidot*thetadot*sin(theta)*cos(theta))/(m*sqr(length)/8);
            //    torque = -10*thetadot+6.1*exp(-5.9*(theta+10*pi/180))-10.5*exp(-21.8*(67*pi/180-theta));
            //    
            //}}
            
            //function rungekutta(thetab,thetadotb,phib,phidotb){{
            //    MotionEquation(thetab,thetadotb,phib,phidotb);
            //    k1=0.5*dt*thetadotdot;
            //    k11=0.5*dt*phidotdot;
            //  
            //    MotionEquation(thetab+0.5*dt*(thetadotb+0.5*k1),thetadotb+k1,phib+0.5*dt*(phidotb+0.5*k11),phidotb+k11);
            //    k2=0.5*dt*thetadotdot;
            //    k21=0.5*dt*phidotdot;
              
            //    MotionEquation(thetab+0.5*dt*(thetadotb+0.5*k1),thetadotb+k2,phib+0.5*dt*(phidotb+0.5*k11),phidotb+k21);
            //    k3=0.5*dt*thetadotdot;
            //    k31=0.5*dt*phidotdot;
              
             //   MotionEquation(thetab+dt*(thetadotb+k3),thetadotb+2*k3,phib+dt*(phidotb+k31),phidotb+2*k31);
             //   k4=0.5*dt*thetadotdot;
             //   k41=0.5*dt*phidotdot;
              
             //   theta=theta+dt*(thetadot+1/3*(k1+k2+k3));
             //   thetadot=thetadot+1/3*(k1+2*k2+2*k3+k4);
              
             //   phi=phi+dt*(phidot+1/3*(k11+k21+k31));
              //  phidot:=phidot+1/3*(k11+2*k21+2*k31+k41);
             //   }}
             
            function drawCube(projection, view, model) {{
            twgl.setBuffersAndAttributes(gl, programInfo, cubeBufferInfo);
            twgl.setUniforms(programInfo, {{
                u_color: color,
                u_lightDir: lightDir,
                u_projection: projection,
                u_view: view,
                u_model: model,
            }});
            twgl.drawBufferInfo(gl, cubeBufferInfo);
            }}
            function drawSphere(projection, view, model) {{
            twgl.setBuffersAndAttributes(gl, programInfo, sphereBufferInfo);
            twgl.setUniforms(programInfo, {{
                u_color: color,
                u_lightDir: lightDir,
                u_projection: projection,
                u_view: view,
                u_model: model,
            }});
            twgl.drawBufferInfo(gl, sphereBufferInfo);
            }}
            function drawCylinder(projection, view, model) {{
            twgl.setBuffersAndAttributes(gl, programInfo, cylinderBufferInfo);
            twgl.setUniforms(programInfo, {{
                u_color: color,
                u_lightDir: lightDir,
                u_projection: projection,
                u_view: view,
                u_model: model,
            }});
            twgl.drawBufferInfo(gl, cylinderBufferInfo);
            }}
            
            function drawCylinder2(projection, view, model) {{
            twgl.setBuffersAndAttributes(gl, programInfo, cylinder2BufferInfo);
            twgl.setUniforms(programInfo, {{
                u_color: color,
                u_lightDir: lightDir,
                u_projection: projection,
                u_view: view,
                u_model: model,
            }});
            twgl.drawBufferInfo(gl, cylinder2BufferInfo);
            }}
            
            function drawCylinder3(projection, view, model) {{
            twgl.setBuffersAndAttributes(gl, programInfo, cylinder3BufferInfo);
            twgl.setUniforms(programInfo, {{
                u_color: color,
                u_lightDir: lightDir,
                u_projection: projection,
                u_view: view,
                u_model: model,
            }});
            twgl.drawBufferInfo(gl, cylinder3BufferInfo);
            }}
            
            function drawCone(projection, view, model) {{
            twgl.setBuffersAndAttributes(gl, programInfo, truncatCylinderBufferInfo);
            twgl.setUniforms(programInfo, {{
                u_color: color,
                u_lightDir: lightDir,
                u_projection: projection,
                u_view: view,
                u_model: model,
            }});
            twgl.drawBufferInfo(gl, truncatCylinderBufferInfo);
            }}
            </script>
            """,
            width=640,
            height=640,
        )
        
        
    def one(self):
        with st.expander("1. Develop motion equation of each plane by Lagrangian Method using single pendulum modeling."):
            st.markdown("## Motion Equation")
            st.latex(
                r"""
                    \begin{aligned}
                    x&=l_p sin{\theta}cos{\varphi}\\
                    y&=l_p sin{\theta}sin{\varphi}\\
                    z &= -l_{p}cos{\theta}\\
                    \end{aligned}
                    """
            )
            st.markdown("### Positions in Cartesian")
            st.latex(
                r"""
                    \begin{aligned}
                    x&=l_psin{\theta}cos{\varphi}\\
                    y&=l_psin{\theta}sin{\varphi}\\
                    z&= -l_pcos{\theta}\\
                    \end{aligned}
                    """
            )
            st.markdown("### Velocity")
            st.latex(
                r"""
                    \begin{aligned}
                    \dot{x}&=l_p(\dot{\theta}cos{\theta}cos{\varphi}-\dot{\varphi}sin{\theta}sin{\varphi})\\
                    \dot{y}&=l_p(\dot{\theta}cos{\theta}sin{\varphi}+\dot{\varphi}sin{\theta}cos{\varphi})\\
                    \dot{z}&=l_p\dot{\theta}sin{\theta}\\
                    \end{aligned}
                    """
            )
            st.markdown("### Velocity Squared")
            st.latex(
                r"""
                    \begin{aligned}
                    \dot{x}^2&=l_p^2((\dot{\theta}cos{\theta}cos{\varphi})^{2}+(\dot{\varphi}sin{\theta}sin{\varphi})^{2}-2\dot{\theta}cos{\theta}cos{\varphi}\dot{\varphi}sin{\theta}sin{\varphi})\\
                    \dot{y}^2&=l_p^2((\dot{\theta}cos{\theta}sin{\varphi})^{2}+(\dot{\varphi}sin{\theta}cos{\varphi})^{2}+2\dot{\theta}cos{\theta}sin{\varphi}\dot{\varphi}sin{\theta}cos{\varphi})\\
                    \dot{z}^2&=l_p^2\dot{\theta^2}sin^2{\theta}\\
                    \\
                    \dot{x}^2+\dot{y}^2+\dot{z}^2&=\quad l_p^2((\dot{\theta}cos{\theta}cos{\varphi})^{2}+(\dot{\varphi}sin{\theta}sin{\varphi})^{2}-2\dot{\theta}cos{\theta}cos{\varphi}\dot{\varphi}sin{\theta}sin{\varphi})\\
                                                    &\quad+l_p^2((\dot{\theta}cos{\theta}sin{\varphi})^{2}+(\dot{\varphi}sin{\theta}cos{\varphi})^{2}+2\dot{\theta}cos{\theta}sin{\varphi}\dot{\varphi}sin{\theta}cos{\varphi})\\
                                                    &\quad+l_p^2\dot{\theta^2}sin^2{\theta}\\
                                                        &=l_p^2(\dot{\theta}^2+\dot{\varphi}^{2}sin^2{\theta})\\\\
                    \end{aligned}
                    """
            )

            st.markdown("### Kinetic Energy and Potential Energy")
            st.latex(
                r"""
                    \begin{aligned}
                    EK&=\frac{1}{2}mv^2+\frac{1}{2}I\dot{\theta}^{2}\\
                    EK&=\frac{1}{2}(mv^2+I\dot{\theta}^{2})\\
                    v^2&=\dot{x}^2+\dot{y}^2+\dot{z}^2\\
                    \\
                    EP&=mgh^2\\
                    EP&=g(mh+m_2h_2)\\
                    h&=0.818H-l_pcos{\theta}\\
                    \end{aligned}
                    """
            )
            st.markdown("### Lagrange Function")
            st.latex(
                r"""
                    \begin{aligned}
                    L&=EK-EP\\
                    L&=\frac{1}{2}(mv^2+I\dot{\theta}^{2})-g(mh)\\
                    L&=\frac{1}{2}(m(l_{p}^2(\dot{\theta}^2+\dot{\varphi}^{2}sin^2{\theta}))-g(m_(0.818H-l_{p}cos{\theta}))\\
                    \end{aligned}
                    """
            )
            st.markdown("### Lagrange Equation")
            st.latex(
                r"""
                    \begin{aligned}
                    \frac{d}{dt} \frac{\partial L}{\partial \dot{\alpha}}-\frac{\partial L}{\partial \alpha}&=\tau\\
                    \\
                    \frac{\partial L}{\partial \theta}&=\frac{1}{2}(2ml_{p}^2\dot{\varphi}^{2}sin{\theta}cos{\theta}-g(ml_{p}sin{\theta}))\\
                    \\\frac{\partial L}{\partial \dot{\theta}}&=\frac{1}{2}(2ml_{p}^2\dot{\theta}+2I\dot{\theta})\\
                    \\
                    \frac{d}{dt} \frac{\partial L}{\partial \dot{\theta}}&=\frac{1}{2}(2ml_{p}^2\ddot{\theta}+2I\ddot{\theta})\\
                        \\
                        \\
                    \frac{d}{dt} \frac{\partial L}{\partial \dot{\theta}}-\frac{\partial L}{\partial \theta}&=
                    \dot{\theta}(0)\\
                    & + \ddot{\theta}(ml_p^2\ddot{\theta}+I\ddot{\theta})\\
                    & + \dot{\varphi}(-ml_p^2\dot{\varphi}sin\theta cos\theta)\\
                    & + \ddot{\varphi}(0)\\
                    & + g(ml_{p}sin{\theta})\\
                    \\\\\\
                    \frac{\partial L}{\partial \varphi}&=0\\
                    \\
                    \frac{\partial L}{\partial \dot{\varphi}}&=\frac{1}{2}(2ml_{p}^2\dot{\varphi}sin^2{\theta})\\
                    \\
                    \frac{d}{dt} \frac{\partial L}{\partial \dot{\varphi}}&=\frac{1}{2}(2ml_{p}^2(\ddot{\varphi}sin^2{\theta}+2\dot{\varphi}\dot{\theta}sin{\theta}cos{\theta})\\
                    \\\\
                    \frac{d}{dt} \frac{\partial L}{\partial \dot{\varphi}}-\frac{\partial L}{\partial \varphi}&=
                    \dot{\theta}(0)\\
                    & + \ddot{\theta}(0)\\
                    & + \dot{\varphi}(2 ml_p^2 \dot{\theta} sin\theta cos\theta)\\
                    & + \ddot{\varphi}(ml_p^2 sin^2 \theta)\\
                    \\\\\\
                    \end{aligned}
                    """
            )
            st.markdown("### Motion Equation")
            st.latex(
                r"""
                    \begin{aligned}
                    \frac{d}{dt} \frac{\partial L}{\partial \dot{\theta}}-\frac{\partial L}{\partial \theta}&=\tau_{\theta}\\
                    \frac{d}{dt} \frac{\partial L}{\partial \dot{\varphi}}-\frac{\partial L}{\partial \varphi}&=\tau_{\varphi}\\
                    \begin{bmatrix}
                    ml_p^2\ddot{\theta}+I\ddot{\theta}&0 \\ 
                    0&ml_p^2 sin^2 \theta 
                   \end{bmatrix}
                   \begin{bmatrix}
                    \ddot{\theta} \\ 
                     \ddot{\varphi} \\
                   \end{bmatrix}+
                   \begin{bmatrix}
                   0&-ml_p^2\dot{\varphi}sin\theta cos\theta\\
                       0&2 ml_p^2 \dot{\theta} sin\theta cos\theta
                   \end{bmatrix}
                   \begin{bmatrix}
                    \dot{\theta} \\ 
                     \dot{\varphi} \\
                   \end{bmatrix}
                   +
                   \begin{bmatrix}
                   g(ml_{p}sin{\theta})\\
                    0
                   \end{bmatrix}
                   &= 
                   \begin{bmatrix}
                   \tau_{\theta}\\
                    \tau_{\varphi}
                   \end{bmatrix}\\
                    \end{aligned}
                    """
            )
            st.markdown("### Runge Kutta Method")
            st.latex(
                r"""
                    \begin{aligned}
                    \ddot{\theta}&= f(t,\theta,\dot{\theta})\\
                    k_1&=\frac{h}{2}f(t,\theta,\dot{\theta})\\
                    k_2&=\frac{h}{2}f\left( t+\frac{h}{2},\theta+\frac{h}{2}\left(\dot{\theta}+\frac{k1}{2}\right),\dot{\theta}+k_1\right)\\
                    k_3&=\frac{h}{2}f\left( t+\frac{h}{2},\theta+\frac{h}{2}\left(\dot{\theta}+\frac{k1}{2}\right),\dot{\theta}+k_2\right)\\
                    k_4&=\frac{h}{2}f\left( t+h,\theta+h\left(\dot{\theta}+k_3\right),\dot{\theta}+2k_3\right)\\
                    \theta_n&=\theta_{n-1}+h\left(\dot{\theta}_{n-1}+\frac{k_1+k_2+k_3}{3}\right)\\
                    \dot{\theta}_n&=\dot{\theta}_{n-1}+h\left(k_1+2k_2+2k_3+k_4\right)\\
                    \\
                        \\
                    \ddot{\varphi}&= f(t,\varphi,\dot{\varphi})\\
                    k_1&=\frac{h}{2}f(t,\varphi,\dot{\varphi})\\
                    k_2&=\frac{h}{2}f\left( t+\frac{h}{2},\varphi+\frac{h}{2}\left(\dot{\varphi}+\frac{k1}{2}\right),\dot{\varphi}+k_1\right)\\
                    k_3&=\frac{h}{2}f\left( t+\frac{h}{2},\varphi+\frac{h}{2}\left(\dot{\varphi}+\frac{k1}{2}\right),\dot{\varphi}+k_2\right)\\
                    k_4&=\frac{h}{2}f\left( t+h,\varphi+h\left(\dot{\varphi}+k_3\right),\dot{\varphi}+2k_3\right)\\
                    \varphi_n&=\varphi_{n-1}+h\left(\dot{\varphi}_{n-1}+\frac{k_1+k_2+k_3}{3}\right)\\
                    \dot{\varphi}_n&=\dot{\varphi}_{n-1}+h\left(k_1+2k_2+2k_3+k_4\right)\\
                    \end{aligned}
                    """
            )

    def two(self):
        with st.expander(
            """
            2. Use the motion equation above to develop a model of Wrist joint movement. Use regression of 
            body segment and anthropometric data with your BW and BH. Moment of inertia at rotation 
            axis can be simplified by simple joint assumption. Assume all the limbs are uniform 
            rod model.Identify passive joint torque for flexor/extensor and abductor/adductor by performing passive movement 
            test for each axis by performing iterative computational experiments. 
            """
        ):
            st.latex(
                r"""\begin{aligned}
                    l_1 &= Upper\;Arm\;Length \\ 
                    l_2 &= Lower\;Arm\;Length \\
                    l_3 &= Hand\;Length \\
                    l_{1p} &= Upper\;Arm\;Proximal\;Length \\ 
                    l_{2p} &= Lower\;Arm\;Proximal\;Length \\
                    l_{3p} &= Hand\;Proximal\;Length \\
                    m_1&= Upper\;Arm\;Mass\\ 
                    m_2&=Lower\;Arm\;Mass\\
                    m_3&=Hand\;Mass\\
                    \\
                    l_1&=17.2\% * H\\
                    l_2&=15.7\% * H\\
                    l_3&=5.75\% * H\\
                    l_{1p}&=43.6\% * l_1\\ 
                    l_{2p}&=43\% * l_2\\
                    l_{3p}&=49.4\% * l_3\\
                    m_1&=2.8\% * M\\
                    m_2&=1.6\% * M\\
                    m_3&=0.6\% * M\\
                    \\
                    H&= 183\;cm\\
                    M&= 94\;kg\\
                    \\
                    l_1&=31.476\;cm\\
                    l_2&=28.731\;cm\\
                    l_3&=10.5225\;cm\\
                    l_{1p}&=13.723536\;cm\\ 
                    l_{2p}&=12.35433\;cm\\
                    l_{3p}&=5.198115\;cm\\
                    m_1&=2.632\;kg\\
                    m_2&=1.504\;kg\\
                    m_3&=0.564\;kg\\
                    \end{aligned}"""
            )
            
            st.markdown("### Motion Equation")
            st.latex(
                r"""
                    \begin{aligned}
                    \frac{d}{dt} \frac{\partial L}{\partial \dot{\theta}}-\frac{\partial L}{\partial \theta}&=\tau_{\theta}\\
                    \frac{d}{dt} \frac{\partial L}{\partial \dot{\varphi}}-\frac{\partial L}{\partial \varphi}&=\tau_{\varphi}\\
                    \begin{bmatrix}
                    ml_p^2\ddot{\theta}+I\ddot{\theta}&0 \\ 
                    0&ml_p^2 sin^2 \theta 
                   \end{bmatrix}
                   \begin{bmatrix}
                    \ddot{\theta} \\ 
                     \ddot{\varphi} \\
                   \end{bmatrix}+
                   \begin{bmatrix}
                   0&-ml_p^2\dot{\varphi}sin\theta cos\theta\\
                       0&2 ml_p^2 \dot{\theta} sin\theta cos\theta
                   \end{bmatrix}
                   \begin{bmatrix}
                    \dot{\theta} \\ 
                     \dot{\varphi} \\
                   \end{bmatrix}
                   +
                   \begin{bmatrix}
                   g(ml_{p}sin{\theta})\\
                    0
                   \end{bmatrix}
                   &= 
                   \begin{bmatrix}
                   \tau_{\theta}\\
                    \tau_{\varphi}
                   \end{bmatrix}\\
                       \\
                           \\
                               \\
                    I_g&=\frac{ml^2}{12}\\
                    I_o&=I_g+m(\frac{l}{2})^2\\
                    \end{aligned}
                    """
            )

    def three(self):
        with st.expander(
            """
            3.Determine the active joint torque to simulate 3d movements by setting reference of each 
            movement as sinusoidal function of each axis. This step we can use PID Control 
            Algorithm. The active joint torques to simulate a certain movement speed in can be 
            concluded by appropriate Kp, Kd, Ki with low RMSE. as performed in the program 
            example. 
            """
        ):
            camera_radius = st.slider("Camera Radius", 0, 500, 135)
            camera_theta = st.slider("Camera Theta", 0, 180, 90)
            camera_phi = st.slider("Camera Phi", 0, 360, 44)
            H = st.number_input("Height", 183, 183, 183)
            M = st.number_input("Mass", 93, 93, 93)
            self.webgl_component(camera_radius, camera_theta, camera_phi, H, M)

        
if __name__ == "__main__":
    st.set_page_config(page_title="Motion Simulation", page_icon="🤖")
    Main().main()