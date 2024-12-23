import numpy as np
import vtk

def generate_aircraft_model():
    # Increase resolution for denser points
    theta = np.linspace(0, 2 * np.pi, 60)
    z_fuse = np.linspace(0, 10, 60)
    theta_fuse, z_fuse = np.meshgrid(theta, z_fuse)
    x_fuse = 1 * np.cos(theta_fuse)
    y_fuse = 1 * np.sin(theta_fuse)
    
    # Increase wing point density
    wing_span = 5
    wing_width = 0.1
    x_wing = np.linspace(-wing_span, wing_span, 10)
    y_wing = np.linspace(-wing_width, wing_width, 4)
    x_wing, y_wing = np.meshgrid(x_wing, y_wing)
    z_wing = np.full_like(x_wing, 5)
    
    x = np.concatenate([x_fuse.flatten(), x_wing.flatten()])
    y = np.concatenate([y_fuse.flatten(), y_wing.flatten()])
    z = np.concatenate([z_fuse.flatten(), z_wing.flatten()])
    return x, y, z

def simulate_airflow(x, y, z):
    # Let airflow magnitude depend on distance from origin
    magnitude = np.sqrt(x**2 + y**2 + z**2) + 0.1
    # Direct airflow in the x-direction, scale by magnitude
    u = magnitude
    v = np.zeros_like(y)
    w = np.zeros_like(z)
    return u, v, w

def visualize_airflow_vtk(x, y, z, u, v, w):
    # Create airflow vectors
    points = vtk.vtkPoints()
    vectors = vtk.vtkFloatArray()
    vectors.SetNumberOfComponents(3)
    vectors.SetName("Vectors")
    
    for xi, yi, zi, ui, vi, wi in zip(x, y, z, u, v, w):
        points.InsertNextPoint(xi, yi, zi)
        vectors.InsertNextTuple3(ui, vi, wi)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().SetVectors(vectors)
    
    arrow = vtk.vtkArrowSource()
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(arrow.GetOutputPort())
    glyph.SetInputData(polydata)
    glyph.SetVectorModeToUseVector()
    # Scale glyphs by vector length
    glyph.SetScaleModeToScaleByVector()
    glyph.SetScaleFactor(0.1)
    glyph.Update()
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())
    
    airflow_actor = vtk.vtkActor()
    airflow_actor.SetMapper(mapper)
    airflow_actor.GetProperty().SetColor(0, 0, 1)
    
    # Simple aircraft model
    aircraft_points = vtk.vtkPoints()
    for xi, yi, zi in zip(x, y, z):
        aircraft_points.InsertNextPoint(xi, yi, zi)
    
    aircraft_polydata = vtk.vtkPolyData()
    aircraft_polydata.SetPoints(aircraft_points)
    
    aircraft_mapper = vtk.vtkPolyDataMapper()
    aircraft_mapper.SetInputData(aircraft_polydata)
    
    aircraft_actor = vtk.vtkActor()
    aircraft_actor.SetMapper(aircraft_mapper)
    aircraft_actor.GetProperty().SetColor(1, 0, 0)
    
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWindow)
    
    renderer.AddActor(airflow_actor)
    renderer.AddActor(aircraft_actor)
    renderer.SetBackground(1, 1, 1)
    
    renderWindow.Render()
    renderInteractor.Start()

def main():
    x, y, z = generate_aircraft_model()
    u, v, w = simulate_airflow(x, y, z)
    visualize_airflow_vtk(x, y, z, u, v, w)

if __name__ == "__main__":
    main()
