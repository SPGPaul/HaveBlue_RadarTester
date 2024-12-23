import tkinter as tk
from tkinter import ttk
import vtk
import numpy as np
import hashlib
import re

class RadarWaveScatteringSimulation:
    def __init__(self, shape='aircraft', size=1.0, frequency=1e10, num_points=1000):
        self.shape = shape
        self.size = size
        self.frequency = frequency
        self.num_points = num_points
        self.wavelength = 3e8 / frequency
        self.setup_renderer()

    def setup_renderer(self):
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window_interactor = vtk.vtkRenderWindowInteractor()
        self.render_window_interactor.SetRenderWindow(self.render_window)

        self.renderer.GradientBackgroundOn()
        self.renderer.SetBackground(0.1, 0.1, 0.4)
        self.renderer.SetBackground2(0.8, 0.8, 1.0)
        self.add_orientation_marker()

        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0, -6, 3)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)

        self.create_shape()
        self.create_incoming_waves()
        self.create_scattered_waves()

        self.render_window.Render()
        self.render_window_interactor.Initialize()
        self.render_window_interactor.Start()

    def add_orientation_marker(self):
        axes = vtk.vtkAxesActor()
        marker = vtk.vtkOrientationMarkerWidget()
        marker.SetOrientationMarker(axes)
        marker.SetInteractor(self.render_window_interactor)
        marker.SetEnabled(True)
        marker.InteractiveOff()

    def create_shape(self):
        if self.shape == 'sphere':
            shape_source = vtk.vtkSphereSource()
            shape_source.SetRadius(self.size)
            shape_source.SetThetaResolution(50)
            shape_source.SetPhiResolution(50)
            shape_mapper = vtk.vtkPolyDataMapper()
            shape_mapper.SetInputConnection(shape_source.GetOutputPort())
            shape_actor = vtk.vtkActor()
            shape_actor.SetMapper(shape_mapper)
            shape_actor.GetProperty().SetColor(1, 0, 0)
            self.renderer.AddActor(shape_actor)
        elif self.shape == 'cube':
            shape_source = vtk.vtkCubeSource()
            shape_source.SetXLength(self.size)
            shape_source.SetYLength(self.size)
            shape_source.SetZLength(self.size)
            shape_mapper = vtk.vtkPolyDataMapper()
            shape_mapper.SetInputConnection(shape_source.GetOutputPort())
            shape_actor = vtk.vtkActor()
            shape_actor.SetMapper(shape_mapper)
            shape_actor.GetProperty().SetColor(1, 0, 0)
            self.renderer.AddActor(shape_actor)
        elif self.shape == 'aircraft':
            self.create_aircraft_shape()

    def create_aircraft_shape(self):
        fuselage = vtk.vtkCylinderSource()
        fuselage.SetRadius(self.size * 0.08)
        fuselage.SetHeight(self.size * 3.5)
        fuselage.SetResolution(50)
        fuselage_mapper = vtk.vtkPolyDataMapper()
        fuselage_mapper.SetInputConnection(fuselage.GetOutputPort())
        fuselage_actor = vtk.vtkActor()
        fuselage_actor.SetMapper(fuselage_mapper)
        fuselage_actor.RotateX(90)

        nose_cone = vtk.vtkConeSource()
        nose_cone.SetRadius(self.size * 0.08)
        nose_cone.SetHeight(self.size * 0.4)
        nose_cone.SetResolution(50)
        nose_cone_mapper = vtk.vtkPolyDataMapper()
        nose_cone_mapper.SetInputConnection(nose_cone.GetOutputPort())
        nose_cone_actor = vtk.vtkActor()
        nose_cone_actor.SetMapper(nose_cone_mapper)
        nose_cone_actor.SetPosition(0, self.size * 1.75, 0)
        nose_cone_actor.RotateX(90)

        cockpit = vtk.vtkSphereSource()
        cockpit.SetRadius(self.size * 0.12)
        cockpit.SetThetaResolution(50)
        cockpit.SetPhiResolution(50)
        cockpit_mapper = vtk.vtkPolyDataMapper()
        cockpit_mapper.SetInputConnection(cockpit.GetOutputPort())
        cockpit_actor = vtk.vtkActor()
        cockpit_actor.SetMapper(cockpit_mapper)
        cockpit_actor.SetPosition(0, self.size * 0.9, 0.15)

        def make_wing(points, thickness=0.02):
            poly = vtk.vtkPolygon()
            poly.GetPointIds().SetNumberOfIds(len(points))
            pts = vtk.vtkPoints()
            for i, p in enumerate(points):
                pts.InsertNextPoint(p)
                poly.GetPointIds().SetId(i, i)
            pd = vtk.vtkPolyData()
            pd.SetPoints(pts)
            polys = vtk.vtkCellArray()
            polys.InsertNextCell(poly)
            pd.SetPolys(polys)
            extrude = vtk.vtkLinearExtrusionFilter()
            extrude.SetInputData(pd)
            extrude.SetExtrusionTypeToNormalExtrusion()
            extrude.SetScaleFactor(thickness)
            extrude.Update()
            return extrude

        main_wing = make_wing([
            (self.size * 1.3, 0, 0.0),
            (0, 0, -self.size * 1.0),
            (-self.size * 1.3, 0, 0.0)
        ])
        main_wing_mapper = vtk.vtkPolyDataMapper()
        main_wing_mapper.SetInputConnection(main_wing.GetOutputPort())
        main_wing_actor = vtk.vtkActor()
        main_wing_actor.SetMapper(main_wing_mapper)
        main_wing_actor.SetPosition(0, 0, self.size * 0.7)

        tail_wing = make_wing([
            (self.size * 0.7, 0, 0),
            (0, 0, -self.size * 0.4),
            (-self.size * 0.7, 0, 0)
        ])
        tail_wing_mapper = vtk.vtkPolyDataMapper()
        tail_wing_mapper.SetInputConnection(tail_wing.GetOutputPort())
        tail_wing_actor = vtk.vtkActor()
        tail_wing_actor.SetMapper(tail_wing_mapper)
        tail_wing_actor.SetPosition(0, 0, -self.size * 1.2)

        stabilizer = make_wing([
            (0, 0, 0),
            (self.size * 0.15, 0, -self.size * 0.1),
            (0, self.size * 0.8, 0)
        ])
        stabilizer_mapper = vtk.vtkPolyDataMapper()
        stabilizer_mapper.SetInputConnection(stabilizer.GetOutputPort())
        stabilizer_actor = vtk.vtkActor()
        stabilizer_actor.SetMapper(stabilizer_mapper)
        stabilizer_actor.SetPosition(0, 0, -self.size * 1.2)
        stabilizer_actor.RotateX(90)

        append_filter = vtk.vtkAppendPolyData()
        append_filter.AddInputConnection(fuselage.GetOutputPort())
        append_filter.AddInputConnection(nose_cone.GetOutputPort())
        append_filter.AddInputConnection(cockpit.GetOutputPort())
        append_filter.AddInputConnection(main_wing.GetOutputPort())
        append_filter.AddInputConnection(tail_wing.GetOutputPort())
        append_filter.AddInputConnection(stabilizer.GetOutputPort())
        append_filter.Update()

        shape_mapper = vtk.vtkPolyDataMapper()
        shape_mapper.SetInputConnection(append_filter.GetOutputPort())
        shape_actor = vtk.vtkActor()
        shape_actor.SetMapper(shape_mapper)
        shape_actor.GetProperty().SetColor(1, 0, 0)
        shape_actor.GetProperty().SetSpecular(0.5)
        shape_actor.GetProperty().SetSpecularPower(30)
        self.renderer.AddActor(shape_actor)

    def create_incoming_waves(self):
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        for i in range(self.num_points):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            x = -self.size * np.sin(phi) * np.cos(theta)
            y = -self.size * np.sin(phi) * np.sin(theta)
            z = -self.size * np.cos(phi)
            tx = np.random.uniform(-self.size, self.size)
            ty = np.random.uniform(-self.size, self.size)
            tz = np.random.uniform(-self.size, self.size)
            points.InsertNextPoint(x, y, z)
            points.InsertNextPoint(tx, ty, tz)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i * 2)
            line.GetPointIds().SetId(1, i * 2 + 1)
            lines.InsertNextCell(line)
        lines_poly_data = vtk.vtkPolyData()
        lines_poly_data.SetPoints(points)
        lines_poly_data.SetLines(lines)
        lines_mapper = vtk.vtkPolyDataMapper()
        lines_mapper.SetInputData(lines_poly_data)
        lines_actor = vtk.vtkActor()
        lines_actor.SetMapper(lines_mapper)
        lines_actor.GetProperty().SetColor(0, 1, 1)
        lines_actor.GetProperty().SetLineWidth(2)
        self.renderer.AddActor(lines_actor)

    def create_scattered_waves(self):
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        for i in range(self.num_points):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            x = -self.size * np.sin(phi) * np.cos(theta)
            y = -self.size * np.sin(phi) * np.sin(theta)
            z = -self.size * np.cos(phi)
            tx = np.random.uniform(-self.size, self.size)
            ty = np.random.uniform(-self.size, self.size)
            tz = np.random.uniform(-self.size, self.size)
            normal = np.array([0, 0, 1])
            incoming = np.array([x, y, z])
            reflection = incoming - 2 * np.dot(incoming, normal) * normal
            reflection *= -1
            points.InsertNextPoint(tx, ty, tz)
            points.InsertNextPoint(reflection[0], reflection[1], reflection[2])
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i * 2)
            line.GetPointIds().SetId(1, i * 2 + 1)
            lines.InsertNextCell(line)
        lines_poly_data = vtk.vtkPolyData()
        lines_poly_data.SetPoints(points)
        lines_poly_data.SetLines(lines)
        lines_mapper = vtk.vtkPolyDataMapper()
        lines_mapper.SetInputData(lines_poly_data)
        lines_actor = vtk.vtkActor()
        lines_actor.SetMapper(lines_mapper)
        lines_actor.GetProperty().SetColor(0, 0, 1)
        lines_actor.GetProperty().SetLineWidth(2)
        self.renderer.AddActor(lines_actor)

    def update_simulation(self, shape=None, size=None, frequency=None, num_points=None):
        if shape:
            self.shape = shape
        if size:
            self.size = size
        if frequency:
            self.frequency = frequency
            self.wavelength = 3e8 / frequency
        if num_points:
            self.num_points = num_points
        self.renderer.RemoveAllViewProps()
        self.create_shape()
        self.create_incoming_waves()
        self.create_scattered_waves()
        self.render_window.Render()

def main():
    root = tk.Tk()
    root.title("Radar Simulation Parameters")
    root.geometry("400x300")
    root.configure(bg="#34495E")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure(".", background="#34495E", foreground="white", font=("Helvetica", 12))
    style.configure("TLabel", background="#34495E", foreground="white", font=("Helvetica", 12))
    style.configure("TButton", background="#1ABC9C", foreground="white", font=("Helvetica", 10, "bold"))
    style.configure("TEntry", fieldbackground="#ffffff", foreground="black")
    style.configure("TFrame", background="#34495E")

    main_frame = ttk.Frame(root, padding="20 20 20 20", style="TFrame")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(main_frame, text="Shape:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    shape_var = tk.StringVar(value="aircraft")
    shape_combo = ttk.Combobox(main_frame, textvariable=shape_var, values=["sphere", "cube", "aircraft"])
    shape_combo.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(main_frame, text="Size:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    size_var = tk.StringVar(value="1.0")
    size_entry = ttk.Entry(main_frame, textvariable=size_var)
    size_entry.grid(row=1, column=1, padx=5, pady=5)

    ttk.Label(main_frame, text="Radar Frequency (Hz):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
    freq_var = tk.StringVar(value="1e10")
    freq_entry = ttk.Entry(main_frame, textvariable=freq_var)
    freq_entry.grid(row=2, column=1, padx=5, pady=5)

    ttk.Label(main_frame, text="Number of Radar Waves:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
    num_points_var = tk.StringVar(value="1000")
    num_points_entry = ttk.Entry(main_frame, textvariable=num_points_var)
    num_points_entry.grid(row=3, column=1, padx=5, pady=5)

    simulation = [None]

    def on_submit():
        shape = shape_var.get()
        size = float(size_var.get())
        frequency = float(freq_var.get())
        num_points = int(num_points_var.get())
        if simulation[0] is None:
            simulation[0] = RadarWaveScatteringSimulation(shape, size, frequency, num_points)
        else:
            simulation[0].update_simulation(shape, size, frequency, num_points)

    def on_cancel():
        root.destroy()

    btn_frame = ttk.Frame(main_frame, style="TFrame")
    btn_frame.grid(row=4, column=0, columnspan=2, pady=10)

    ttk.Button(btn_frame, text="Submit", command=on_submit).grid(row=0, column=0, padx=5)
    ttk.Button(btn_frame, text="Cancel", command=on_cancel).grid(row=0, column=1, padx=5)

    root.mainloop()
    users = {"admin": hashlib.sha256("admin".encode()).hexdigest()}

    def hash_password(password):
        return hashlib.sha256(password.encode()).hexdigest()

    def is_password_safe(password):
        if len(password) < 8:
            return False
        if not re.search(r"[A-Z]", password):
            return False
        if not re.search(r"[a-z]", password):
            return False
        if not re.search(r"[0-9]", password):
            return False
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return False
        return True

    def login():
        login_window = tk.Toplevel(root)
        login_window.title("Login")
        login_window.geometry("300x200")
        login_window.configure(bg="#34495E")

        ttk.Label(login_window, text="Username:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        username_var = tk.StringVar()
        username_entry = ttk.Entry(login_window, textvariable=username_var)
        username_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(login_window, text="Password:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        password_var = tk.StringVar()
        password_entry = ttk.Entry(login_window, textvariable=password_var, show="*")
        password_entry.grid(row=1, column=1, padx=5, pady=5)

        def on_login():
            username = username_var.get()
            password = password_var.get()
            hashed_password = hash_password(password)
            if username in users and users[username] == hashed_password:
                login_window.destroy()
                main()
            else:
                ttk.Label(login_window, text="Invalid credentials", foreground="red").grid(row=3, column=0, columnspan=2, pady=5)

        ttk.Button(login_window, text="Login", command=on_login).grid(row=2, column=0, columnspan=2, pady=10)

    def register():
        register_window = tk.Toplevel(root)
        register_window.title("Register")
        register_window.geometry("300x250")
        register_window.configure(bg="#34495E")

        ttk.Label(register_window, text="Username:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        username_var = tk.StringVar()
        username_entry = ttk.Entry(register_window, textvariable=username_var)
        username_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(register_window, text="Password:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        password_var = tk.StringVar()
        password_entry = ttk.Entry(register_window, textvariable=password_var, show="*")
        password_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(register_window, text="Confirm Password:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        confirm_password_var = tk.StringVar()
        confirm_password_entry = ttk.Entry(register_window, textvariable=confirm_password_var, show="*")
        confirm_password_entry.grid(row=2, column=1, padx=5, pady=5)

        def on_register():
            username = username_var.get()
            password = password_var.get()
            confirm_password = confirm_password_var.get()
            if password != confirm_password:
                ttk.Label(register_window, text="Passwords do not match", foreground="red", background="#34495E").grid(row=4, column=0, columnspan=2, pady=5)
                return
            if not is_password_safe(password):
                ttk.Label(register_window, text="Password is not safe", foreground="red").grid(row=4, column=0, columnspan=2, pady=5)
                return
            users[username] = hash_password(password)
            register_window.destroy()

        ttk.Button(register_window, text="Register", command=on_register).grid(row=3, column=0, columnspan=2, pady=10)

    root = tk.Tk()
    root.title("User Authentication")
    root.geometry("300x200")
    root.configure(bg="#34495E")

    ttk.Button(root, text="Login", command=login).pack(pady=10)
    ttk.Button(root, text="Register", command=register).pack(pady=10)

    root.mainloop()
    if __name__ == "__main__":
        main()

    def calculate_reflection_percentage(self):
        ground_hits = 0
        for i in range(self.num_points):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            x = -self.size * np.sin(phi) * np.cos(theta)
            y = -self.size * np.sin(phi) * np.sin(theta)
            z = -self.size * np.cos(phi)
            normal = np.array([0, 0, 1])
            incoming = np.array([x, y, z])
            reflection = incoming - 2 * np.dot(incoming, normal) * normal
            reflection *= -1
            if reflection[2] < 0:  # Check if the z-component of the reflection vector is negative
                ground_hits += 1
        percentage = (ground_hits / self.num_points) * 100
        self.display_reflection_percentage(percentage)

    def display_reflection_percentage(self, percentage):
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(f"Ground Hit Percentage: {percentage:.2f}%")
        text_actor.GetTextProperty().SetFontSize(24)
        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        text_actor.SetPosition(10, self.render_window.GetSize()[1] - 40)
        self.renderer.AddActor2D(text_actor)
        self.render_window.Render()

    def update_simulation(self, shape=None, size=None, frequency=None, num_points=None):
        if shape:
            self.shape = shape
        if size:
            self.size = size
        if frequency:
            self.frequency = frequency
            self.wavelength = 3e8 / frequency
        if num_points:
            self.num_points = num_points
        self.renderer.RemoveAllViewProps()
        self.create_shape()
        self.create_incoming_waves()
        self.create_scattered_waves()
        self.render_window.Render()
        self.calculate_reflection_percentage()
