import vtk

# --- Wczytanie danych DICOM ---
reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName("C:/Users/jacek/OneDrive/Pulpit/inf_med_3/lab3/mr_brainixA")  # katalog z plikami DICOM
reader.Update()
imageData = reader.GetOutput()

# --- Kontur: izopowierzchnia ---
contour = vtk.vtkContourFilter()
contour.SetInputConnection(reader.GetOutputPort())
contour.SetValue(0, 100)  # wartość iso (później modyfikowana suwakiem)

# --- Kolorowanie powierzchni ---
colorFunc = vtk.vtkColorTransferFunction()
colorFunc.AddRGBPoint(0, 0.0, 0.0, 1.0)
colorFunc.AddRGBPoint(500, 1.0, 1.0, 0.0)
colorFunc.AddRGBPoint(1000, 1.0, 0.0, 0.0)

# --- Mapper (3D) ---
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(contour.GetOutputPort())
mapper.SetLookupTable(colorFunc)
mapper.SetScalarRange(imageData.GetScalarRange())

# --- Aktor 3D ---
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# --- Renderer i okno ---
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.1, 0.1, 0.2)
renderer.AddActor(actor)

renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindow.SetSize(800, 600)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renderWindow)

# --- Slider (zmiana wartości izopowierzchni) ---
class IsoCallback:
    def __init__(self, contour, renderWindow):
        self.contour = contour
        self.renderWindow = renderWindow
    def __call__(self, caller, event):
        value = caller.GetSliderRepresentation().GetValue()
        self.contour.SetValue(0, value)
        self.renderWindow.Render()

sliderRep = vtk.vtkSliderRepresentation2D()
sliderRep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
sliderRep.GetPoint1Coordinate().SetValue(0.1, 0.1)
sliderRep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
sliderRep.GetPoint2Coordinate().SetValue(0.4, 0.1)
sliderRep.SetMinimumValue(0)
sliderRep.SetMaximumValue(1000)
sliderRep.SetValue(100)
sliderRep.SetTitleText("Iso Value")

sliderWidget = vtk.vtkSliderWidget()
sliderWidget.SetInteractor(interactor)
sliderWidget.SetRepresentation(sliderRep)
sliderWidget.SetAnimationModeToAnimate()
sliderWidget.EnabledOn()
sliderWidget.AddObserver("InteractionEvent", IsoCallback(contour, renderWindow))


style = vtk.vtkInteractorStyleJoystickCamera()
interactor.SetInteractorStyle(style)
# --- Start interakcji ---
interactor.Initialize()
renderWindow.Render()
interactor.Start()
