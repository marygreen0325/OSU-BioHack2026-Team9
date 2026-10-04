import tkinter as tk
from tkinter import ttk
import math
import random
import time
from collections import deque

from matplotlib import style
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class FeedingTubeOcclusionSimulation:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Feeding Tube Occlusion Detection & External Vibration Simulation"
        )

        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)

        
        ## SIMULATION STATE
        

        self.running = False

        ## Actual simulated obstruction level.
        ## 0.0 = clear
        ## 1.0 = completely obstructed

        self.occlusion = 0.0

        ## Target obstruction level used when a test event
        ## is injected.

        self.target_occlusion = 0.0

        ## External actuator state

        self.vibration_active = False

        ## Conceptual actuator intensity

        self.vibration_intensity = 0.65

        ## Detection state

        self.occlusion_detected = False

        self.warning_level = "NORMAL"

        self.detection_counter = 0

        self.detection_confidence = 0.0

        ## Simulation clock

        self.sim_time = 0.0

        
        ## OCCLUSION SENSOR MODEL
        

        ## Baseline sensor signal when tube is clear

        self.sensor_baseline = 5.0

        ## Conceptual response of sensor to obstruction

        self.sensor_response_gain = 105.0

        ## Simulated measurement noise

        self.sensor_noise = 2.0

        ## Sensor detection threshold

        self.sensor_warning_threshold = 30.0

        ## Strong detection threshold

        self.sensor_critical_threshold = 75.0

        ## Number of consecutive abnormal samples required

        self.required_detection_samples = 8

        ## Current raw sensor signal

        self.occlusion_sensor_signal = 0.0

        ## Filtered sensor signal

        self.filtered_sensor_signal = 0.0

        
        ## ACTUATOR MODEL
        

        ## Arbitrary simulation parameter.
        ##
        ## This is NOT a real clinical vibration/clearance value.

        self.clearance_rate = 0.08

        
        ## HISTORY
        

        self.history_length = 250

        self.time_history = deque(
            maxlen=self.history_length
        )

        self.sensor_history = deque(
            maxlen=self.history_length
        )

        self.occlusion_history = deque(
            maxlen=self.history_length
        )

        self.vibration_history = deque(
            maxlen=self.history_length
        )

        
        ## TUBE VISUALIZATION
        

        self.tube_left = 70
        self.tube_right = 940

        self.tube_top = 145
        self.tube_bottom = 265

        ## Occlusion sensor is near beginning of tube

        self.occlusion_sensor_x = 180

        ## External actuator is also near beginning

        self.actuator_x = 220

        self.actuator_y = self.tube_bottom + 75

        ## Obstruction is downstream

        self.clog_x = 700

        ## Vibration animation

        self.vibration_phase = 0.0

        ## Flow particles are purely visual

        self.particles = []

        
        ## BUILD APPLICATION
        

        self.build_interface()

        self.initialize_particles()

        self.last_update = time.time()

        self.update_simulation()

 
    ## BUILD INTERFACE


    def build_interface(self):

        background_color = "#edf4fa"
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "TFrame",
            background=background_color
        )
        style.configure(
            "TLabelframe",
            background=background_color,
            bordercolor="#243a5e",
            lightcolor="#243a5e",
            darkcolor="#243a5e",
            borderwidth=2
        )
        style.configure(
            "TLabelframe.Label",
            background=background_color
        )
        style.configure(
            "TLabel",
            background=background_color
        )
        style.configure(
            "TButton",
            background="#cfe3f1",
            bordercolor="#243a5e",
            lightcolor="#243a5e",
            darkcolor="#243a5e",
            borderwidth=2
        )
        style.map(
            "TButton",
            background=[
                ("pressed", "#cfe3f1"),
                ("active", "#cfe3f1")
            ]
        )
        self.root.configure(
            background=background_color
        )

        
        ## TITLE
        

        title_frame = ttk.Frame(
            self.root
        )

        title_frame.pack(
            fill="x",
            padx=15,
            pady=(12, 5)
        )

        title = ttk.Label(
            title_frame,
            text=(
                "Feeding Tube Occlusion Detection "
                "& External Vibration Simulation"
            ),
            font=("Times New Roman", 20, "bold")
        )

        title.pack(
            anchor="w"
        )

        subtitle = ttk.Label(
            title_frame,
            text=(
                "Conceptual closed-loop system: "
                "occlusion sensing → detection → "
                "external actuator response"
            ),
            font=("Times New Roman", 10)
        )

        subtitle.pack(
            anchor="w",
            pady=(3, 0)
        )

        tabbed_area = ttk.Frame(self.root)

        tabbed_area.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )

        tabbed_area.grid_rowconfigure(0, weight=1)
        tabbed_area.grid_columnconfigure(0, weight=1)

        page_container = ttk.Frame(tabbed_area)

        page_container.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        page_container.grid_rowconfigure(0, weight=1)
        page_container.grid_columnconfigure(0, weight=1)

        self.tab_bar = tk.Frame(
            tabbed_area,
            background="#dce6f1",
            padx=3,
            pady=3
        )

        self.tab_bar.grid(
            row=1,
            column=0,
            sticky="ew"
        )

        simulation_page = ttk.Frame(page_container)

        simulation_page.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        self.pages = {
            "Simulation": simulation_page
        }

        
        ## SCROLLABLE MAIN AREA
        

        outer_frame = ttk.Frame(
            simulation_page
        )

        outer_frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )

        self.scroll_canvas = tk.Canvas(
            outer_frame,
            highlightthickness=0
        )

        self.scroll_canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar = ttk.Scrollbar(
            outer_frame,
            orient="vertical",
            command=self.scroll_canvas.yview
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        self.scroll_canvas.configure(
            yscrollcommand=scrollbar.set
        )

        main = ttk.Frame(
            self.scroll_canvas
        )

        self.scroll_window = self.scroll_canvas.create_window(
            (0, 0),
            window=main,
            anchor="nw"
        )

        def update_scroll_region(event=None):

            self.scroll_canvas.configure(
                scrollregion=self.scroll_canvas.bbox("all")
            )

        main.bind(
            "<Configure>",
            update_scroll_region
        )

        def resize_scroll_frame(event):

            self.scroll_canvas.itemconfigure(
                self.scroll_window,
                width=event.width
            )

        self.scroll_canvas.bind(
            "<Configure>",
            resize_scroll_frame
        )

        self.scroll_canvas.bind_all(
            "<MouseWheel>",
            self.on_mousewheel
        )

        
        ## STATUS CARDS
        

        status_frame = ttk.Frame(
            main
        )

        status_frame.pack(
            fill="x",
            padx=5,
            pady=5
        )

        self.sensor_value = self.create_status_card(
            status_frame,
            "OCCLUSION SENSOR",
            "0.0",
            0
        )

        self.occlusion_value = self.create_status_card(
            status_frame,
            "SIMULATED OCCLUSION",
            "0%",
            1
        )

        self.confidence_value = self.create_status_card(
            status_frame,
            "DETECTION CONFIDENCE",
            "0%",
            2
        )

        self.vibration_value = self.create_status_card(
            status_frame,
            "ACTUATOR",
            "OFF",
            3
        )

        self.state_value = self.create_status_card(
            status_frame,
            "SYSTEM STATE",
            "NORMAL",
            4,
            colored=True
        )

        
        ## SIMULATION CONTROLS
        

        controls = ttk.LabelFrame(
            main,
            text="Simulation Controls",
            padding=10
        )

        controls.pack(
            fill="x",
            padx=5,
            pady=8
        )

        button_row = ttk.Frame(
            controls
        )

        button_row.pack(
            fill="x"
        )

        self.start_button = ttk.Button(
            button_row,
            text="▶ Start Simulation",
            command=self.toggle_running
        )

        self.start_button.pack(
            side="left",
            padx=3
        )

        ttk.Button(
            button_row,
            text="Mild Occlusion",
            command=lambda: self.inject_occlusion(0.35)
        ).pack(
            side="left",
            padx=3
        )

        ttk.Button(
            button_row,
            text="Moderate Occlusion",
            command=lambda: self.inject_occlusion(0.60)
        ).pack(
            side="left",
            padx=3
        )

        ttk.Button(
            button_row,
            text="Severe Occlusion",
            command=lambda: self.inject_occlusion(0.85)
        ).pack(
            side="left",
            padx=3
        )

        ttk.Button(
            button_row,
            text="Near Complete",
            command=lambda: self.inject_occlusion(0.98)
        ).pack(
            side="left",
            padx=3
        )

        ttk.Button(
            button_row,
            text="Reset",
            command=self.reset_simulation
        ).pack(
            side="left",
            padx=3
        )

        ttk.Button(
            button_row,
            text="Toggle Vibration",
            command=self.toggle_vibration
        ).pack(
            side="left",
            padx=3
        )

        
        ## ACTUATOR INTENSITY
        

        intensity_frame = ttk.Frame(
            controls
        )

        intensity_frame.pack(
            fill="x",
            pady=(10, 0)
        )

        ttk.Label(
            intensity_frame,
            text="Conceptual actuator intensity:"
        ).pack(
            side="left"
        )

        self.intensity_var = tk.DoubleVar(
            value=self.vibration_intensity
        )

        self.intensity_slider = ttk.Scale(
            intensity_frame,
            from_=0.0,
            to=1.0,
            variable=self.intensity_var,
            command=self.update_intensity
        )

        self.intensity_slider.pack(
            side="left",
            fill="x",
            expand=True,
            padx=10
        )

        self.intensity_label = ttk.Label(
            intensity_frame,
            text="65%"
        )

        self.intensity_label.pack(
            side="left"
        )

        
        ## REAL-TIME TUBE VISUALIZATION
        

        visual_frame = ttk.LabelFrame(
            main,
            text="Real-Time Tube Visualization",
            padding=10
        )

        visual_frame.pack(
            fill="x",
            padx=5,
            pady=8
        )
        self.tube_canvas = tk.Canvas(
            visual_frame,
            height=470,
            background="white",
            highlightthickness=1,
            highlightbackground="#243a5e"
        )

        self.tube_canvas.pack(
            side="left",
            fill="both",
            expand=True,
        )

        
        ## CLOSED LOOP LOGIC
        

        logic_frame = ttk.LabelFrame(
            visual_frame,
            text="Closed-Loop System Logic",
            padding=12
        )

        logic_frame.pack(
            side="right",
            fill="y",
            padx=(12, 0)
        )

        self.logic_labels = {}

        logic_items = [

            (
                "Occlusion sensor",
                "MONITORING"
            ),

            (
                "Signal processing",
                "WAITING"
            ),

            (
                "Occlusion detection",
                "CLEAR"
            ),

            (
                "Controller",
                "IDLE"
            ),

            (
                "External actuator",
                "OFF"
            )

        ]

        for name, state in logic_items:

            row = ttk.Frame(
                logic_frame
            )

            row.pack(
                fill="x",
                pady=5
            )

            ttk.Label(
                row,
                text=name,
                width=22
            ).pack(
                side="left"
            )

            label = ttk.Label(
                row,
                text=state
            )

            label.pack(
                side="right"
            )

            self.logic_labels[name] = label

        ttk.Separator(
            logic_frame
        ).pack(
            fill="x",
            pady=12
        )

        ttk.Label(
            logic_frame,
            text="DETECTION CONFIDENCE",
            font=("Times New Roman", 9, "bold")
        ).pack(
            anchor="w"
        )

        self.confidence_bar = ttk.Progressbar(
            logic_frame,
            orient="horizontal",
            length=220,
            mode="determinate",
            maximum=100
        )

        self.confidence_bar.pack(
            fill="x",
            pady=6
        )

        self.confidence_label = ttk.Label(
            logic_frame,
            text="0%"
        )

        self.confidence_label.pack(
            anchor="e"
        )

        ttk.Label(
            logic_frame,
            text=(
                "\nCONTROL SEQUENCE\n\n"
                "1. Occlusion sensor continuously samples\n"
                "2. Sensor signal is filtered\n"
                "3. Abnormal signal is detected\n"
                "4. Detection confidence increases\n"
                "5. Controller activates actuator\n"
                "6. Simulated obstruction decreases\n"
                "7. Sensor confirms clearance"
            ),
            justify="left"
        ).pack(
            anchor="w",
            pady=10
        )

        
        ## LIVE GRAPHS TAB

        graph_frame = ttk.Frame(
            page_container,
            padding=12
        )

        graph_frame.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        self.pages["Live Graphs"] = graph_frame

        log_frame = ttk.Frame(
            page_container,
            padding=10
        )

        log_frame.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        self.pages["System Event Log"] = log_frame

        self.figure, self.axes = plt.subplots(
            3,
            1,
            figsize=(12, 6),
            sharex=True
        )

        self.figure.subplots_adjust(
            left=0.08,
            right=0.97,
            top=0.96,
            bottom=0.08,
            hspace=0.50
        )

        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=graph_frame
        )

        self.graph_widget = self.canvas.get_tk_widget()

        self.graph_widget.pack(
            fill="both",
            expand=True
        )

        self.log_text = tk.Text(
            log_frame,
            height=18,
            wrap="word",
            state="disabled",
            font=("Courier New", 9)
        )

        self.log_text.pack(
            side="left",
            fill="both",
            expand=True
        )

        log_scrollbar = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.log_text.yview
        )

        log_scrollbar.pack(
            side="right",
            fill="y"
        )

        self.log_text.configure(
            yscrollcommand=log_scrollbar.set
        )

        self.tab_buttons = {}

        def select_tab(name):

            self.pages[name].tkraise()

            for tab_name, button in self.tab_buttons.items():
                button.configure(
                    relief=(
                        tk.SUNKEN
                        if tab_name == name
                        else tk.RAISED
                    ),
                    background=(
                        "#cfe3f1"
                        if tab_name == name
                        else "#90e0ef"
                    ),
                    activebackground=(
                        "#cfe3f1"
                        if tab_name == name
                        else "#90e0ef"
                    )
                )

        for name in self.pages:

            button = tk.Button(
                self.tab_bar,
                text=name,
                font=("Segoe UI", 10, "bold"),
                padx=20,
                pady=9,
                relief=tk.RAISED,
                background="#90e0ef",
                activebackground="#90e0ef",
                command=lambda tab_name=name: select_tab(tab_name)
            )

            button.pack(
                side="left",
                fill="x",
                expand=True,
                padx=2
            )

            self.tab_buttons[name] = button

        select_tab("Simulation")

        self.log(
            "SYSTEM INITIALIZED"
        )

        self.log(
            "Continuous occlusion sensing active"
        )

        self.log(
            "External vibration actuator ready"
        )

        self.log(
            "Simulation mode: conceptual / non-clinical"
        )

        self.draw_tube()
        self.draw_particles()
        self.draw_obstruction()
        self.draw_vibration()

    
    ## STATUS CARD
    

    def create_status_card(
        self,
        parent,
        title,
        value,
        column,
        colored=False
    ):

        frame = ttk.LabelFrame(
            parent,
            text=title,
            padding=8
        )

        frame.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=4
        )

        parent.columnconfigure(
            column,
            weight=1
        )

        if colored:
            label = tk.Label(
                frame,
                text=value,
                font=("Times New Roman", 14, "bold"),
                padx=4,
                pady=2
            )
        else:
            label = ttk.Label(
                frame,
                text=value,
                font=("Times New Roman", 14, "bold")
            )

        label.pack()

        return label

    
    ## MOUSE WHEEL
    

    def on_mousewheel(
        self,
        event
    ):

        self.scroll_canvas.yview_scroll(
            int(-1 * (event.delta / 120)),
            "units"
        )

    
    ## INITIALIZE FLOW PARTICLES
    

    def initialize_particles(self):

        self.particles = []

        for _ in range(120):

            self.particles.append({

                "x": random.uniform(
                    self.tube_left,
                    self.tube_right
                ),

                "y": random.uniform(
                    self.tube_top + 12,
                    self.tube_bottom - 12
                ),

                "speed": random.uniform(
                    1.0,
                    2.8
                ),

                "size": random.uniform(
                    2.5,
                    5.5
                ),

                "phase": random.uniform(
                    0,
                    math.pi * 2
                )

            })

    
    ## DRAW TUBE
    

    def draw_tube(self):

        c = self.tube_canvas

        c.delete("tube")

        
        ## MAIN TUBE
        

        c.create_rectangle(
            self.tube_left,
            self.tube_top,
            self.tube_right,
            self.tube_bottom,
            fill="#f4f4f4",
            outline="#555555",
            width=3,
            tags="tube"
        )

        ## Inner tube

        c.create_rectangle(
            self.tube_left + 5,
            self.tube_top + 5,
            self.tube_right - 5,
            self.tube_bottom - 5,
            fill="#fafafa",
            outline="",
            tags="tube"
        )

        
        ## FLOW DIRECTION
        

        ## Raised well above the sensor.

        flow_arrow_y = (
            self.tube_top - 105
        )

        c.create_line(
            self.tube_left,
            flow_arrow_y,
            self.tube_right,
            flow_arrow_y,
            arrow=tk.LAST,
            width=2,
            fill="#555555",
            tags="tube"
        )

        c.create_text(
            self.tube_left,
            flow_arrow_y - 18,
            text="FLOW DIRECTION →",
            anchor="w",
            font=("Times New Roman", 9, "bold"),
            fill="#555555",
            tags="tube"
        )

        
        ## OCCLUSION SENSOR
        

        sensor_top = (
            self.tube_top - 85
        )

        sensor_bottom = (
            self.tube_top - 35
        )

        c.create_rectangle(
            self.occlusion_sensor_x - 65,
            sensor_top,
            self.occlusion_sensor_x + 65,
            sensor_bottom,
            fill="#b1d4eb",
            outline="#1f5f8b",
            width=2,
            tags="tube"
        )

        c.create_text(
            self.occlusion_sensor_x,
            (sensor_top + sensor_bottom) / 2,
            text="OCCLUSION\nSENSOR",
            font=("Times New Roman", 9, "bold"),
            fill="white",
            justify="center",
            tags="tube"
        )

        ## Sensor connection

        c.create_line(
            self.occlusion_sensor_x,
            sensor_bottom,
            self.occlusion_sensor_x,
            self.tube_top,
            fill="#b1d4eb",
            width=4,
            tags="tube"
        )

        
        ## EXTERNAL VIBRATION ACTUATOR
        

        actuator_top = (
            self.tube_bottom + 35
        )

        actuator_bottom = (
            actuator_top + 45
        )

        c.create_rectangle(
            self.actuator_x - 75,
            actuator_top,
            self.actuator_x + 75,
            actuator_bottom,
            fill="#e3911e",
            outline="#a05d00",
            width=2,
            tags="tube"
        )

        c.create_text(
            self.actuator_x,
            (actuator_top + actuator_bottom) / 2,
            text="EXTERNAL\nVIBRATION ACTUATOR",
            font=("Times New Roman", 9, "bold"),
            fill="white",
            justify="center",
            tags="tube"
        )

        ## Actuator connection

        c.create_line(
            self.actuator_x,
            actuator_top,
            self.actuator_x,
            self.tube_bottom,
            fill="#e3911e",
            width=4,
            tags="tube"
        )

        
        ## LEGEND
        

        legend_y = 440

        c.create_text(
            80,
            legend_y,
            text="𖹭 Simulated flow particles",
            fill="#f29f9f",
            anchor="w",
            font=("Times New Roman", 9),
            tags="tube"
        )

        c.create_text(
            240,
            legend_y,
            text="𖹭 Occlusion sensor",
            fill="#b1d4eb",
            anchor="w",
            font=("Times New Roman", 9),
            tags="tube"
        )

        c.create_text(
            365,
            legend_y,
            text="𖹭 External vibration actuator",
            fill="#e3911e",
            anchor="w",
            font=("Times New Roman", 9),
            tags="tube"
        )

        c.create_text(
            540,
            legend_y,
            text="𖹭 Simulated obstruction",
            fill="#8b5e3c",
            anchor="w",
            font=("Times New Roman", 9),
            tags="tube"
        )

    
    ## DRAW FLOW PARTICLES
    

    def draw_particles(self):

        c = self.tube_canvas

        c.delete("particle")

        for particle in self.particles:

            y = (
                particle["y"]
                + math.sin(
                    self.sim_time * 2
                    + particle["phase"]
                ) * 2.5
            )

            c.create_oval(
                particle["x"] - particle["size"],
                y - particle["size"],
                particle["x"] + particle["size"],
                y + particle["size"],
                fill="#f29f9f",
                outline="",
                tags="particle"
            )

    
    ## UPDATE FLOW PARTICLES
    

    def update_particles(self):

        for particle in self.particles:

            speed = particle["speed"]

            distance_to_clog = (
                self.clog_x
                - particle["x"]
            )

            ## Visual slowing near obstruction

            if (
                0 < distance_to_clog < 140
                and self.occlusion > 0.15
            ):

                slowdown = (
                    1
                    - self.occlusion * 0.85
                )

                speed *= max(
                    0.08,
                    slowdown
                )

            ## Very severe obstruction

            if (
                particle["x"] >= self.clog_x
                and self.occlusion > 0.75
            ):

                speed *= 0.08

            particle["x"] += speed

            ## Wrap particles around

            if particle["x"] > self.tube_right:

                particle["x"] = self.tube_left

                particle["y"] = random.uniform(
                    self.tube_top + 12,
                    self.tube_bottom - 12
                )

    
    ## DRAW OBSTRUCTION
    

    def draw_obstruction(self):

        c = self.tube_canvas

        c.delete("obstruction")

        if self.occlusion <= 0.01:

            return

        
        ## OBSTRUCTION DIMENSIONS
        

        tube_height = (
            self.tube_bottom
            - self.tube_top
        )

        ## Keep obstruction completely inside tube.

        max_height = (
            tube_height * 0.72
        )

        height = (
            8
            + self.occlusion * max_height
        )

        width = (
            25
            + self.occlusion * 80
        )

        
        ## POSITION
        

        left = (
            self.clog_x
            - width / 2
        )

        right = (
            self.clog_x
            + width / 2
        )

        ## Keep bottom inside tube.

        bottom = (
            self.tube_bottom - 7
        )

        top = (
            bottom - height
        )

        ## Keep top inside tube.

        top = max(
            self.tube_top + 7,
            top
        )

        
        ## MAIN OBSTRUCTION
        

        c.create_oval(
            left,
            top,
            right,
            bottom,
            fill="#8b5e3c",
            outline="#5f3f29",
            width=2,
            tags="obstruction"
        )

        
        ## TEXTURE
        

        random.seed(42)

        pieces = int(
            5
            + self.occlusion * 15
        )

        for _ in range(pieces):

            x = random.uniform(
                left + 4,
                right - 4
            )

            y = random.uniform(
                top + 4,
                bottom - 4
            )

            size = random.uniform(
                3,
                8
            )

            ## Keep pieces inside tube.

            x = max(
                self.tube_left + 5,
                min(
                    self.tube_right - 5,
                    x
                )
            )

            y = max(
                self.tube_top + 5,
                min(
                    self.tube_bottom - 5,
                    y
                )
            )

            c.create_oval(
                x - size,
                y - size,
                x + size,
                y + size,
                fill="#70452e",
                outline="",
                tags="obstruction"
            )

        
        ## LABEL
        

        c.create_text(
            self.clog_x,
            top - 15,
            text=(
                f"SIMULATED OBSTRUCTION "
                f"{self.occlusion * 100:.0f}%"
            ),
            font=("Times New Roman", 9, "bold"),
            fill="#7c2d12",
            tags="obstruction"
        )

    
    ## DRAW VIBRATION
    

    def draw_vibration(self):

        c = self.tube_canvas

        c.delete("vibration")

        if not self.vibration_active:

            return

        self.vibration_phase += 0.35

        ## Expanding vibration waves

        for i in range(4):

            radius = (
                15
                + (
                    (
                        self.vibration_phase
                        + i * 18
                    ) % 80
                )
            )

            c.create_oval(
                self.actuator_x - radius,
                self.actuator_y - radius,
                self.actuator_x + radius,
                self.actuator_y + radius,
                outline="#f1d6a9",
                width=2,
                tags="vibration"
            )

        ## Small vibration lines

        for i in range(6):

            x = (
                self.actuator_x
                - 35
                + i * 12
            )

            wave = (
                math.sin(
                    self.vibration_phase + i
                ) * 5
            )

            c.create_line(
                x,
                self.tube_bottom + 5,
                x + 6,
                self.tube_bottom + 5 + wave,
                fill="#f4d5af",
                width=2,
                tags="vibration"
            )

    
    ## OCCLUSION SENSOR
    

    def calculate_occlusion_sensor(self):

        ## Convert simulated obstruction into
        ## a conceptual sensor signal.

        signal = (
            self.sensor_baseline
            + self.sensor_response_gain
            * self.occlusion
        )

        ## Add simulated sensor noise

        signal += random.gauss(
            0,
            self.sensor_noise
        )

        signal = max(
            0,
            min(
                100,
                signal
            )
        )

        self.occlusion_sensor_signal = signal

        ## Low-pass filter

        alpha = 0.15

        self.filtered_sensor_signal = (
            alpha
            * self.occlusion_sensor_signal
            + (
                1 - alpha
            )
            * self.filtered_sensor_signal
        )

    
    ## DETECTION LOGIC
    

    def update_detection(self):

        sensor_problem = (
            self.filtered_sensor_signal
            > self.sensor_warning_threshold
        )

        ## Require several abnormal samples

        if sensor_problem:

            self.detection_counter += 1

        else:

            self.detection_counter = max(
                0,
                self.detection_counter - 1
            )

        if (
            self.filtered_sensor_signal <= self.sensor_warning_threshold
        ):
            self.detection_confidence = 0.0
        else: 
            signal_range = (
                self.sensor_critical_threshold
                - self.sensor_warning_threshold
            )
            signal_confidence = (
                self.filtered_sensor_signal
                - self.sensor_warning_threshold
            ) / signal_range
            signal_confidence = max(
                0.0,
                min(1.0, signal_confidence)
            )

            persistence_confidence = min(
                1.0,
                self.detection_counter
                / self.required_detection_samples
            )
        ## both a strong signal and persistence are required for high confidence

            self.detection_confidence = (
                signal_confidence
                *persistence_confidence
            )

        
        ## CONFIRMED OCCLUSION

        activation_threshold = 0.65

        if ( 
            self.detection_confidence >= activation_threshold
            and self.detection_counter >= self.required_detection_samples
        ):

            if not self.occlusion_detected:

                self.occlusion_detected = True

                self.log(
                    "OCCLUSION DETECTED: "
                    "persistent abnormal sensor signal"
                )

            if not self.vibration_active:

                self.vibration_active = True

                self.log(
                    "ACTUATOR ACTIVATED: "
                    "external vibration applied"
                )

        
        ## WARNING LEVEL
        

        if (
            self.filtered_sensor_signal
            >= self.sensor_critical_threshold
        ):

            self.warning_level = "CRITICAL"

        elif (
            self.filtered_sensor_signal
            >= self.sensor_warning_threshold
        ):

            self.warning_level = "WARNING"

        else:

            self.warning_level = "NORMAL"


       
        ## CLEARANCE CONFIRMATION
        

        if (
            self.occlusion_detected
            and self.occlusion < 0.08
            and self.filtered_sensor_signal < 20
        ):

            self.occlusion_detected = False

            self.vibration_active = False

            self.target_occlusion = 0.0

            self.detection_counter = 0

            self.log(
                "CLEARANCE CONFIRMED: "
                "occlusion sensor returned toward baseline"
            )

    
    ## EXTERNAL ACTUATOR
    

    def apply_vibration(self, dt):

        if not self.vibration_active:

            return

        intensity = (
            self.vibration_intensity
        )

        ## Arbitrary simulation parameter.
        ##
        ## This does NOT represent a real clinical
        ## actuator setting.

        clearance = (
            self.clearance_rate
            * intensity
            * dt
        )

        variation = random.uniform(
            0.7,
            1.15
        )

        clearance *= variation

        self.occlusion -= clearance

        self.occlusion = max(
            0,
            self.occlusion
        )

        ## Important:
        ##
        ## Once the actuator is clearing the obstruction,
        ## the target follows the actual obstruction instead
        ## of pulling it back toward the original injected value.

        self.target_occlusion = (
            self.occlusion
        )

    
    ## MAIN SIMULATION LOOP
    

    def update_simulation(self):

        now = time.time()

        dt = min(
            0.1,
            now - self.last_update
        )

        self.last_update = now

        if self.running:

            self.sim_time += dt

            ## --------------------------------------------------------
            ## DEVELOP SIMULATED OCCLUSION
            ## --------------------------------------------------------

            ## Before detection/actuation, the obstruction
            ## develops toward the injected test level.

            if not self.vibration_active:

                difference = (
                    self.target_occlusion
                    - self.occlusion
                )

                self.occlusion += (
                    difference
                    * min(
                        1,
                        dt * 0.7
                    )
                )

            ## --------------------------------------------------------
            ## APPLY ACTUATOR
            ## --------------------------------------------------------

            self.apply_vibration(
                dt
            )

            self.occlusion = max(
                0,
                min(
                    1,
                    self.occlusion
                )
            )

            ## --------------------------------------------------------
            ## SENSOR
            ## --------------------------------------------------------

            self.calculate_occlusion_sensor()

            ## --------------------------------------------------------
            ## DETECTION
            ## --------------------------------------------------------

            self.update_detection()

            ## --------------------------------------------------------
            ## PARTICLES
            ## --------------------------------------------------------

            self.update_particles()

            ## --------------------------------------------------------
            ## HISTORY
            ## --------------------------------------------------------

            self.time_history.append(
                self.sim_time
            )

            self.sensor_history.append(
                self.filtered_sensor_signal
            )

            self.occlusion_history.append(
                self.occlusion * 100
            )

            self.vibration_history.append(
                100
                if self.vibration_active
                else 0
            )

            ## --------------------------------------------------------
            ## UI
            ## --------------------------------------------------------

            self.update_status_cards()

            self.update_logic_panel()

            self.update_graphs()

        ## Always redraw visual simulation

        self.draw_particles()

        self.draw_obstruction()

        self.draw_vibration()

        self.root.after(
            50,
            self.update_simulation
        )

    
    ## UPDATE STATUS CARDS
    

    def update_status_cards(self):

        self.sensor_value.config(
            text=(
                f"{self.filtered_sensor_signal:.1f}"
            )
        )

        self.occlusion_value.config(
            text=(
                f"{self.occlusion * 100:.1f}%"
            )
        )

        self.confidence_value.config(
            text=(
                f"{self.detection_confidence * 100:.0f}%"
            )
        )

        if self.vibration_active:

            self.vibration_value.config(
                text=(
                    f"ON "
                    f"({self.vibration_intensity * 100:.0f}%)"
                )
            )

        else:

            self.vibration_value.config(
                text="OFF"
            )

        ## Determine system state

        if self.warning_level == "CRITICAL":

            state = "CRITICAL"

        elif self.warning_level == "WARNING":

            state = "WARNING"

        elif self.vibration_active:

            state = "CLEARING"

        elif self.occlusion_detected:

            state = "OCCLUSION"

        elif self.occlusion >= 0.40:

            state = "DEVELOPING"

        else:

            state = "NORMAL"

        state_colors = {
            "NORMAL": "#4caf50",
            "WARNING": "#ff9800",
            "CRITICAL": "#f44336",
            "CLEARING": "#00acc1",
        }

        state_colors = {
            "NORMAL": "#4caf50",
            "WARNING": "#ff9800",
            "CRITICAL": "#f44336",
        }

        self.state_value.config(
            text=state,
            bg=state_colors.get(state, "#9e9e9e"),
            fg="white"
        )

    
    ## UPDATE LOGIC PANEL
    

    def update_logic_panel(self):

        ## Sensor

        self.logic_labels[
            "Occlusion sensor"
        ].config(
            text=(
                "ABNORMAL"
                if self.filtered_sensor_signal
                > self.sensor_warning_threshold
                else "NORMAL"
            )
        )

        ## Signal processing

        if (
            self.filtered_sensor_signal
            > self.sensor_warning_threshold
        ):

            processing = "ABNORMAL"

        else:

            processing = "STABLE"

        self.logic_labels[
            "Signal processing"
        ].config(
            text=processing
        )

        ## Detection

        self.logic_labels[
            "Occlusion detection"
        ].config(
            text=(
                "DETECTED"
                if self.occlusion_detected
                else "CLEAR"
            )
        )

        ## Controller

        self.logic_labels[
            "Controller"
        ].config(
            text=(
                "ACTIVE"
                if self.vibration_active
                else "IDLE"
            )
        )

        ## Actuator

        self.logic_labels[
            "External actuator"
        ].config(
            text=(
                "VIBRATING"
                if self.vibration_active
                else "OFF"
            )
        )

        ## Confidence

        confidence = (
            self.detection_confidence
            * 100
        )

        self.confidence_bar[
            "value"
        ] = confidence

        self.confidence_label.config(
            text=f"{confidence:.0f}%"
        )

    
    ## UPDATE GRAPHS
    

    def update_graphs(self):

        if len(self.time_history) < 2:

            return

        times = list(
            self.time_history
        )

        sensor = list(
            self.sensor_history
        )

        occlusions = list(
            self.occlusion_history
        )

        vibrations = list(
            self.vibration_history
        )

        
        ## GRAPH 1: SENSOR
        

        self.axes[0].clear()

        self.axes[0].plot(
            times,
            sensor,
            linewidth=2
        )

        self.axes[0].axhline(
            self.sensor_warning_threshold,
            linestyle="--",
            linewidth=1
        )

        self.axes[0].axhline(
            self.sensor_critical_threshold,
            linestyle="--",
            linewidth=1
        )

        self.axes[0].set_ylabel(
            "Sensor Signal"
        )

        self.axes[0].set_title(
            "Occlusion Sensor Signal"
        )

        self.axes[0].set_ylim(
            0,
            105
        )

        self.axes[0].grid(
            alpha=0.25
        )

        
        ## GRAPH 2: OCCLUSION
        

        self.axes[1].clear()

        self.axes[1].plot(
            times,
            occlusions,
            linewidth=2
        )

        self.axes[1].set_ylabel(
            "Occlusion %"
        )

        self.axes[1].set_title(
            "Simulated Obstruction Level"
        )

        self.axes[1].set_ylim(
            0,
            105
        )

        self.axes[1].grid(
            alpha=0.25
        )

        
        ## GRAPH 3: ACTUATOR
        

        self.axes[2].clear()

        self.axes[2].plot(
            times,
            vibrations,
            linestyle="--",
            linewidth=2
        )

        self.axes[2].set_ylabel(
            "Actuator"
        )

        self.axes[2].set_xlabel(
            "Simulation Time (s)"
        )

        self.axes[2].set_title(
            "External Actuator Activity"
        )

        self.axes[2].set_ylim(
            0,
            105
        )

        self.axes[2].grid(
            alpha=0.25
        )

        self.figure.canvas.draw_idle()

    
    ## START / PAUSE
    

    def toggle_running(self):

        self.running = not self.running

        if self.running:

            self.start_button.config(
                text="⏸ Pause Simulation"
            )

            self.log(
                "SIMULATION STARTED"
            )

        else:

            self.start_button.config(
                text="▶ Start Simulation"
            )

            self.log(
                "SIMULATION PAUSED"
            )

    
    ## INJECT OCCLUSION
    

    def inject_occlusion(
        self,
        amount
    ):

        self.target_occlusion = amount

        self.log(
            f"TEST EVENT: Injected "
            f"{amount * 100:.0f}% simulated occlusion"
        )

        if not self.running:

            self.running = True

            self.start_button.config(
                text="⏸ Pause Simulation"
            )

    
    ## MANUAL VIBRATION
    

    def toggle_vibration(self):

        self.vibration_active = (
            not self.vibration_active
        )

        if self.vibration_active:

            self.log(
                "MANUAL OVERRIDE: "
                "external actuator activated"
            )

        else:

            self.log(
                "MANUAL OVERRIDE: "
                "external actuator stopped"
            )

    
    ## ACTUATOR INTENSITY
    

    def update_intensity(
        self,
        value
    ):

        self.vibration_intensity = float(
            value
        )

        self.intensity_label.config(
            text=(
                f"{self.vibration_intensity * 100:.0f}%"
            )
        )

    
    ## RESET
    

    def reset_simulation(self):

        self.running = False

        self.start_button.config(
            text="▶ Start Simulation"
        )

        self.occlusion = 0.0

        self.target_occlusion = 0.0

        self.vibration_active = False

        self.occlusion_detected = False

        self.warning_level = "NORMAL"

        self.detection_counter = 0

        self.occlusion_sensor_signal = 0.0

        self.filtered_sensor_signal = 0.0

        self.detection_confidence = 0.0

        self.time_history.clear()

        self.sensor_history.clear()

        self.occlusion_history.clear()

        self.vibration_history.clear()

        self.sim_time = 0

        self.initialize_particles()

        self.log(
            "SYSTEM RESET"
        )

        self.update_status_cards()

        self.update_logic_panel()

        self.draw_particles()

        self.draw_obstruction()

        self.draw_vibration()

    
    ## EVENT LOG
    

    def log(
        self,
        message
    ):

        timestamp = time.strftime(
            "%H:%M:%S"
        )

        self.log_text.config(
            state="normal"
        )

        self.log_text.insert(
            "end",
            f"[{timestamp}] {message}\n"
        )

        self.log_text.see(
            "end"
        )

        self.log_text.config(
            state="disabled"
        )



## RUN APPLICATION


if __name__ == "__main__":

    root = tk.Tk()

    app = FeedingTubeOcclusionSimulation(
        root
    )

    root.mainloop()