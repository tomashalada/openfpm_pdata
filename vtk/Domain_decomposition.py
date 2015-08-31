#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'

paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Legacy VTK Reader'
dom_boxvtk = LegacyVTKReader(FileNames=['/home/i-bird/Desktop/MOSAIC/OpenFPM_project/OpenFPM_pdata/vtk/CartDecomposition/dom_box.vtk'])

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

renderView1.CameraPosition = [0.5, 0.5, 2.7320508075688776]
renderView1.CameraFocalPoint = [0.5, 0.5, 0.0]
renderView1.CameraParallelScale = 0.5843857695756589

# show data in view
dom_boxvtkDisplay = Show(dom_boxvtk, renderView1)

WriteImage("domain.png")

# change representation type
dom_boxvtkDisplay.SetRepresentationType('Surface With Edges')

WriteImage("domain_decomposed.png")

# create a new 'Legacy VTK Reader'
vtk_partitionvtk = LegacyVTKReader(FileNames=['/home/i-bird/Desktop/MOSAIC/OpenFPM_project/OpenFPM_pdata/vtk/Metis/vtk_partition.vtk'])


idLUT = GetColorTransferFunction('id')
idLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 1.5, 0.865003, 0.865003, 0.865003, 3.0, 0.705882, 0.0156863, 0.14902]
idLUT.ScalarRangeInitialized = 1.0

# show data in view
vtk_partitionvtkDisplay = Show(vtk_partitionvtk, renderView1)
# trace defaults for the display properties.
vtk_partitionvtkDisplay.ColorArrayName = ['POINTS', 'id']
vtk_partitionvtkDisplay.LookupTable = idLUT


#changing interaction mode based on data extents
renderView1.CameraPosition = [0.5, 0.5, 2.7320508075688776]
renderView1.CameraFocalPoint = [0.5, 0.5, 0.0]
renderView1.CameraParallelScale = 0.5843857695756589

# show color bar/color legend
vtk_partitionvtkDisplay.SetScalarBarVisibility(renderView1, False)

# create a new 'Transform'
transform1 = Transform(Input=vtk_partitionvtk)
transform1.Transform = 'Transform'

# Properties modified on transform1.Transform
transform1.Transform.Translate = [0.025, 0.025, 0.0]
transform1.Transform.Scale = [1.0, 1.0, 0.0]

# show data in view
transform1Display = Show(transform1, renderView1)

# hide data in view
Hide(vtk_partitionvtk, renderView1)

# set scalar coloring
ColorBy(transform1Display, ('POINTS', 'computation'))

# change representation type
transform1Display.SetRepresentationType('Points')

# Properties modified on transform1Display
transform1Display.PointSize = 5.0

# set active source
SetActiveSource(vtk_partitionvtk)

# write decomposed
WriteImage("domain_graph.png")

# set scalar coloring
ColorBy(transform1Display, ('POINTS', 'id'))

# change representation type
dom_boxvtkDisplay.SetRepresentationType('Surface')

# write decomposed
WriteImage("domain_graph_decomposed.png")

# create a new 'Legacy VTK Reader'
out_subdomains_0vtk = LegacyVTKReader(FileNames=['/home/i-bird/Desktop/MOSAIC/OpenFPM_project/OpenFPM_pdata/vtk/CartDecomposition/out_subdomains_0.vtk'])

# create a new 'Legacy VTK Reader'
out_subdomains_1vtk = LegacyVTKReader(FileNames=['/home/i-bird/Desktop/MOSAIC/OpenFPM_project/OpenFPM_pdata/vtk/CartDecomposition/out_subdomains_1.vtk'])

# create a new 'Legacy VTK Reader'
out_subdomains_2vtk = LegacyVTKReader(FileNames=['/home/i-bird/Desktop/MOSAIC/OpenFPM_project/OpenFPM_pdata/vtk/CartDecomposition/out_subdomains_2.vtk'])

# create a new 'Legacy VTK Reader'
out_subdomains_3vtk = LegacyVTKReader(FileNames=['/home/i-bird/Desktop/MOSAIC/OpenFPM_project/OpenFPM_pdata/vtk/CartDecomposition/out_subdomains_3.vtk'])

# set active source
#SetActiveSource(out_subdomains_0vtk)

# get color transfer function/color map for 'data'
#dataLUT = GetColorTransferFunction('data')
#dataLUT.RGBPoints = [285.3226244096321, 0.231373, 0.298039, 0.752941, 312.3676762575527, 0.865003, 0.865003, 0.865003, 339.4127281054734, 0.705882, 0.0156863, 0.14902]
#dataLUT.ScalarRangeInitialized = 1.0

# show data in view
out_subdomains_3vtkDisplay = Show(out_subdomains_3vtk, renderView1)
# trace defaults for the display properties.
#out_subdomains_3vtkDisplay.ColorArrayName = ['CELLS', 'data']
#out_subdomains_3vtkDisplay.LookupTable = dataLUT
#out_subdomains_3vtkDisplay.ScalarOpacityUnitDistance = 0.5153778246772277

# show color bar/color legend
#out_subdomains_3vtkDisplay.SetScalarBarVisibility(renderView1, True)

# show data in view
out_subdomains_1vtkDisplay = Show(out_subdomains_1vtk, renderView1)
# trace defaults for the display properties.
#out_subdomains_1vtkDisplay.ColorArrayName = ['CELLS', 'data']
#out_subdomains_1vtkDisplay.LookupTable = dataLUT
#out_subdomains_1vtkDisplay.ScalarOpacityUnitDistance = 0.5153778246772277

# show color bar/color legend
#out_subdomains_1vtkDisplay.SetScalarBarVisibility(renderView1, True)

# show data in view
out_subdomains_2vtkDisplay = Show(out_subdomains_2vtk, renderView1)
# trace defaults for the display properties.
#out_subdomains_2vtkDisplay.ColorArrayName = ['CELLS', 'data']
#out_subdomains_2vtkDisplay.LookupTable = dataLUT
#out_subdomains_2vtkDisplay.ScalarOpacityUnitDistance = 0.5153778246772277

# show color bar/color legend
#out_subdomains_2vtkDisplay.SetScalarBarVisibility(renderView1, True)

# show data in view
out_subdomains_0vtkDisplay = Show(out_subdomains_0vtk, renderView1)
# trace defaults for the display properties.
#out_subdomains_0vtkDisplay.ColorArrayName = ['CELLS', 'data']
#out_subdomains_0vtkDisplay.LookupTable = dataLUT
#out_subdomains_0vtkDisplay.ScalarOpacityUnitDistance = 0.5153778246772277

# show color bar/color legend
#out_subdomains_0vtkDisplay.SetScalarBarVisibility(renderView1, True)

# get opacity transfer function/opacity map for 'data'
#dataPWF = GetOpacityTransferFunction('data')
#dataPWF.Points = [285.3226244096321, 0.0, 0.5, 0.0, 339.4127281054734, 1.0, 0.5, 0.0]
#dataPWF.ScalarRangeInitialized = 1

# turn off scalar coloring
ColorBy(out_subdomains_0vtkDisplay, None)

# set active source
#SetActiveSource(out_subdomains_1vtk)

# turn off scalar coloring
ColorBy(out_subdomains_1vtkDisplay, None)

# turn off scalar coloring
ColorBy(out_subdomains_2vtkDisplay, None)

# set active source
#SetActiveSource(out_subdomains_3vtk)

# turn off scalar coloring
ColorBy(out_subdomains_3vtkDisplay, None)

# change representation type
out_subdomains_3vtkDisplay.SetRepresentationType('Surface With Edges')

# change representation type
out_subdomains_2vtkDisplay.SetRepresentationType('Surface With Edges')

# change representation type
out_subdomains_1vtkDisplay.SetRepresentationType('Surface With Edges')

# change representation type
out_subdomains_0vtkDisplay.SetRepresentationType('Surface With Edges')

# change solid color
out_subdomains_0vtkDisplay.DiffuseColor = [1.0, 0.3333333333333333, 0.0]

# change solid color
out_subdomains_1vtkDisplay.DiffuseColor = [0.6666666666666666, 1.0, 0.0]

# change solid color
out_subdomains_2vtkDisplay.DiffuseColor = [0.0, 0.3333333333333333, 1.0]

# change solid color
out_subdomains_3vtkDisplay.DiffuseColor = [1.0, 1.0, 0.0]

# Properties modified on out_subdomains_0vtkDisplay
out_subdomains_0vtkDisplay.LineWidth = 4.0

# change representation type
out_subdomains_0vtkDisplay.SetRepresentationType('Surface With Edges')

# Properties modified on out_subdomains_1vtkDisplay
out_subdomains_1vtkDisplay.LineWidth = 4.0

# change representation type
out_subdomains_1vtkDisplay.SetRepresentationType('Surface With Edges')

# Properties modified on out_subdomains_2vtkDisplay
out_subdomains_2vtkDisplay.LineWidth = 4.0

# change representation type
out_subdomains_2vtkDisplay.SetRepresentationType('Surface With Edges')

# Properties modified on out_subdomains_3vtkDisplay
out_subdomains_3vtkDisplay.LineWidth = 4.0

# change representation type
out_subdomains_3vtkDisplay.SetRepresentationType('Surface With Edges')

# hide data in view
Hide(dom_boxvtk, renderView1)

WriteImage("domain_subdomain_decomposed.png")

# hide data in view
Hide(transform1, renderView1)

WriteImage("domain_subdomain_decomposed_wg.png")


#Destroy everything
Delete(transform1)
del transform1
Delete(out_subdomains_3vtk)
del out_subdomains_3vtk
Delete(out_subdomains_2vtk)
del out_subdomains_2vtk
Delete(out_subdomains_1vtk)
del out_subdomains_1vtk
Delete(out_subdomains_0vtk)
del out_subdomains_0vtk
Delete(dom_boxvtk)
del dom_boxvtk
Delete(vtk_partitionvtk)
del vtk_partitionvtk