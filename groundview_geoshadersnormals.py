#!

# This is statement is required by the build system to query build info
if __name__ == '__build__':
	raise Exception

import string
__version__ = string.split('$Revision: 1.1.1.1 $')[1]
__date__ = string.join(string.split('$Date: 2007/02/15 19:25:21 $')[1:3], ' ')
__author__ = 'Tarn Weisner Burton <twburton@users.sourceforge.net>'

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
import sys
import math
import numpy
import json
from osgeo import gdal
import random

# Some api in the chain is translating the keystrokes to this octal string
# so instead of saying: ESCAPE = 27, we use the following.
ESCAPE = '\033'
PLUS = '\053'
MINUS = '\055'
A = '\141'
Z = '\172'
R = '\162'
I = '\151'
J = '\152'
K = '\153'
L = '\154'

# Number of the glut window.
window = 0
ridgemode = 0

W = 1.0
H = 1.0

theta = 0.0

phi = 0.0

zlevel = 1.0
xlevel = 0
ylevel = 0

viewangle = 45.0

Y0 = 0
X0 = 0
Y1 = 8111
X1 = 8111
XCENTER = numpy.round((X0+X1)/2)
YCENTER = numpy.round((Y0+Y1)/2)
SCALEFACTOR = 2000.0

WRITEOUT = 1
GEOSHADERS = 0
FAST_REFINING = 1

#WGS84 parameters
a = 6378137.0
e2 = 0.00669437999014

SQSIZE = 8100
LONMIN = -74.500740741
LATMAX = 40.500740741
INITIAL_SQUARES = 100
ROUGHNESS_THRESHOLD = 10

def split(rectIndex, x, y):
	global pointsSet, rectList
	oldRect = rectList[rectIndex]
	del rectList[rectIndex]
	nPoint1 = (oldRect[0][0],y)
	nPoint2 = (x,oldRect[1][1])
	nPoint3 = (oldRect[1][0],y)
	nPoint4 = (x,oldRect[0][1])
	rectList.append(((oldRect[0][0],oldRect[0][1]),(x,y)))
	rectList.append(((x,y),(oldRect[1][0],oldRect[1][1])))
	rectList.append((nPoint1,nPoint2))
	rectList.append((nPoint4,nPoint3))
	pointsSet.add((x,y))
	pointsSet.add(nPoint1)
	pointsSet.add(nPoint2)
	pointsSet.add(nPoint3)
	pointsSet.add(nPoint4)

# A general OpenGL initialization function.  Sets all of the initial parameters. 
def InitGL(Width, Height):				# We call this right after our OpenGL window is created.
	global W, H, viewangle
	W = Width
	H = Height
	glClearColor(0.0, 0.0, 0.0, 0.0)	# This Will Clear The Background Color To Black
	glClearDepth(1.0)					# Enables Clearing Of The Depth Buffer
	glDepthFunc(GL_LESS)				# The Type Of Depth Test To Do
	glEnable(GL_DEPTH_TEST)				# Enables Depth Testing
	glShadeModel(GL_SMOOTH)				# Enables Smooth Color Shading
	geo = gdal.Open('ned19_n47x25_w123x00_wa_puget_sound_2000.img');
	arr = geo.ReadAsArray();
	dim = arr.shape;
	hmax = numpy.amax(arr[X0:(X1+1),Y0:(Y1+1)])
	hmin = numpy.amin(arr[X0:(X1+1),Y0:(Y1+1)])
	global xc, yc, zc
	
	h = arr[XCENTER,YCENTER];
	lon = math.radians(LONMIN+float(YCENTER)/SQSIZE);
	lat = math.radians(LATMAX-float(XCENTER)/SQSIZE);
	n = a/numpy.sqrt(1-e2*(math.sin(lat))**2)
	xc = (n+h)*math.cos(lat)*math.cos(lon)
	yc = (n+h)*math.cos(lat)*math.sin(lon)
	zc = (n*(1-e2)+h)*math.sin(lat)
	print(xc,yc,zc)
	global LIGHT
	norm = numpy.sqrt(xc**2+yc**2+zc**2)
	LIGHT = [xc/norm,yc/norm,zc/norm]

	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()					# Reset The Projection Matrix
	glRotatef(90,-1.0,0,0);
	glRotatef(math.degrees(-math.pi/2+math.atan2(zc, numpy.sqrt(xc**2+yc**2))), -yc, xc, 0);
	matrixthing=glGetFloatv(GL_PROJECTION_MATRIX)
	cameraview=matrixthing[0:3,0:3]
	
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()									# Calculate The Aspect Ratio Of The Window
	gluPerspective(viewangle, float(Width)/float(Height), 0.1, 100.0)
	glMatrixMode(GL_MODELVIEW)

	if GEOSHADERS == 1:
		geofragment = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(geofragment, 'varying vec4 vertex_color; void main() { gl_FragColor = vertex_color; }')
		glCompileShader(geofragment)
		compstat = glGetShaderiv(geofragment,GL_COMPILE_STATUS)
		if (compstat == GL_TRUE):
			print('good')
		geovertex = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(geovertex, 'void main() { gl_Position = ftransform(); }'); 
		glCompileShader(geovertex);
		compstat = glGetShaderiv(geovertex,GL_COMPILE_STATUS)
		if (compstat == GL_TRUE):
			print('good')
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 'varying vec3 vertex_color; void main() {'\
		       'gl_FragColor = vec4(vertex_color, 1); }')
	glCompileShader(fragment)
	compstat = glGetShaderiv(fragment,GL_COMPILE_STATUS)
	if (compstat == GL_TRUE):
		print('good')
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 'varying vec3 vertex_color; uniform vec3 light; void main() {'\
		       'gl_Position = ftransform(); vertex_color = dot(light,gl_Normal)*gl_Color.xyz; }'); 
	glCompileShader(vertex);
	compstat = glGetShaderiv(vertex,GL_COMPILE_STATUS)
	if (compstat == GL_TRUE):
		print('good')
	global shaders;
	shaders = glCreateProgram();
	glAttachShader(shaders,vertex)
	glAttachShader(shaders,fragment)
	glLinkProgram(shaders)
	linkstat = glGetProgramiv(shaders,GL_LINK_STATUS)
	if (linkstat == GL_TRUE):
		print('good')
	else:
		print('bad')
	info = glGetProgramInfoLog(shaders)
	print(info)
	program = '#version 150 \n #extension GL_EXT_geometry_shader4 : enable \n varying vec4 vertex_color;'\
	'layout(triangles_adjacency) in; layout(line_strip, max_vertices=6) out;'\
	'float IsFront(vec3 A, vec3 B, vec3 C){'\
	'float area = (B.x-A.x)*(C.y-A.y)-(C.x-A.x)*(B.y-A.y); return area;}'\
	'void main(){ vec4 v0 = gl_in[0].gl_Position;'\
	'vec4 v1 = gl_in[1].gl_Position;'\
	'vec4 v2 = gl_in[2].gl_Position;'\
	'vec4 v3 = gl_in[3].gl_Position;'\
	'vec4 v4 = gl_in[4].gl_Position;'\
	'vec4 v5 = gl_in[5].gl_Position;'\
	'float c1 = IsFront(v1.w*v2.w*v0.xyz,v0.w*v2.w*v1.xyz,v0.w*v1.w*v2.xyz);'\
	'float c2 = 0;'\
	'float c3 = 0;'\
	'float c4 = 0;'\
	'if (v3==v1) { gl_Position = v1; vertex_color = vec4(1,1,1,1); EmitVertex();'\
	'gl_Position = v2; vertex_color = vec4(1,1,1,1); EmitVertex(); EndPrimitive();}'\
	'else {c2 = IsFront(v3.w*v2.w*v1.xyz,v1.w*v2.w*v3.xyz,v1.w*v3.w*v2.xyz);}'\
	'if (v4==v0) { gl_Position = v0; vertex_color = vec4(1,1,1,1); EmitVertex();'\
	'gl_Position = v2; vertex_color = vec4(1,1,1,1); EmitVertex(); EndPrimitive();}'\
	'else {c3 = IsFront(v2.w*v4.w*v0.xyz,v0.w*v4.w*v2.xyz,v0.w*v2.w*v4.xyz);}'\
	'if (v5==v0) { gl_Position = v0; vertex_color = vec4(1,1,1,1); EmitVertex();'\
	'gl_Position = v1; vertex_color = vec4(1,1,1,1); EmitVertex(); EndPrimitive();}'\
	'else {c4 = IsFront(v5.w*v1.w*v0.xyz,v1.w*v0.w*v5.xyz,v0.w*v5.w*v1.xyz);}'\
	'if (c1 > 0) {'\
	'if (c2 < 0) { gl_Position = v1; vertex_color = vec4(0,0,1,1); EmitVertex();'\
	'gl_Position = v2; vertex_color = vec4(0,0,1,1); EmitVertex();EndPrimitive();}'\
	'if (c3 < 0) {gl_Position = v0; vertex_color = vec4(0,0,1,1); EmitVertex();'\
	'gl_Position = v2; vertex_color = vec4(0,0,1,1); EmitVertex(); EndPrimitive();}'\
	'if (c4 < 0) { gl_Position = v0; vertex_color = vec4(0,0,1,1); EmitVertex();'\
	'gl_Position = v1; vertex_color = vec4(0,0,1,1); EmitVertex();EndPrimitive();}}}'
	if GEOSHADERS == 1:
		ridgeshader = glCreateShader(GL_GEOMETRY_SHADER)
		glShaderSource(ridgeshader,program);
		glCompileShader(ridgeshader);
		compstat = glGetShaderiv(ridgeshader,GL_COMPILE_STATUS)
		glLineWidth(GLfloat(2.0))
		if (compstat == GL_TRUE):
			print('good')
		else:
			print('ridge failed')
		info = glGetShaderInfoLog(ridgeshader)
		print(info)
		global ridgeprogram
		ridgeprogram = glCreateProgram();
		glAttachShader(ridgeprogram,geovertex);
		glAttachShader(ridgeprogram,geofragment);
		glAttachShader(ridgeprogram,ridgeshader);
		glLinkProgram(ridgeprogram);
		linkstat = glGetProgramiv(ridgeprogram,GL_LINK_STATUS)
		if (linkstat == GL_TRUE):
			print('good')
		else:
			print('bad')
		info = glGetProgramInfoLog(ridgeprogram)
		print(info)

	global pointsSet, rectList
	pointsSet = set()
	rectList=[]
	maxdim = max(X1-X0, Y1-Y0);

	for x in range(INITIAL_SQUARES):
		for y in range(INITIAL_SQUARES):
			x1 = X0+numpy.round(float(x)*(X1-X0)/INITIAL_SQUARES)
			y1 = Y0+numpy.round(float(y)*(Y1-Y0)/INITIAL_SQUARES)
			x2 = X0+numpy.round(float(x+1)*(X1-X0)/INITIAL_SQUARES)
			y2 = Y0+numpy.round(float(y+1)*(Y1-Y0)/INITIAL_SQUARES)
			if (x1<X1) and (y1<Y1): 
				x2 = min(x2,X1)
				y2 = min(y2,Y1)
				rectList.append(((x1,y1),(x2,y2)))
				pointsSet.add((x1,y1))
				pointsSet.add((x2,y2))
				pointsSet.add((x1,y2))
				pointsSet.add((x2,y1))
			
	x=0
	while x<len(rectList):
		x1 = rectList[x][0][0]
		y1 = rectList[x][0][1]
		x2 = rectList[x][1][0]
		y2 = rectList[x][1][1]
		if (len(rectList)%1001 ==0):
			print(len(rectList))
		if (x2-x1>1) and (y2-y1>1):
			box = arr[x1:(x2+1),y1:(y2+1)]
			if FAST_REFINING == 0:
				leftupper = box[0,0]
				leftlower = box[x2-x1,0]
				rightupper = box[0,y2-y1]
				rightlower = box[x2-x1,y2-y1]
				xdim = box.shape[0]-1
				ydim = box.shape[1]-1
				if (leftupper > leftlower+rightupper-rightlower):
					linarr = numpy.array([[max(xx*(leftlower-leftupper)/xdim+yy*(rightupper-leftupper)/ydim+leftupper,
							  (xdim-xx)*(-rightlower+rightupper)/xdim+(ydim-yy)*(-rightlower+leftlower)/ydim+rightlower)
							       for yy in range(ydim+1)] for xx in range(xdim+1)])
				else:
					linarr = numpy.array([[min(xx*(leftlower-leftupper)/xdim+yy*(rightupper-leftupper)/ydim+leftupper,
							  (xdim-xx)*(-rightlower+rightupper)/xdim+(ydim-yy)*(-rightlower+leftlower)/ydim+rightlower)
						       for yy in range(ydim+1)] for xx in range(xdim+1)])
				diff1 = max(abs(numpy.amin(linarr-box)),abs(numpy.amax(linarr-box)));
				if (diff1 >= ROUGHNESS_THRESHOLD):
					diff1 = max(abs(numpy.amin(linarr[1:,...]-box[0:-1,...])),abs(numpy.amax(linarr[1:,...]-box[0:-1,...])))
				if (diff1 >= ROUGHNESS_THRESHOLD):
					diff1 = max(abs(numpy.amin(linarr[0:-1,...]-box[1:,...])),abs(numpy.amax(linarr[0:-1,...]-box[1:,...])))
				if (diff1 >= ROUGHNESS_THRESHOLD):
					diff1 = max(abs(numpy.amin(linarr[...,0:-1]-box[...,1:])),abs(numpy.amax(linarr[...,0:-1]-box[...,1:])))
				if (diff1 >= ROUGHNESS_THRESHOLD):
					diff1 = max(abs(numpy.amin(linarr[...,1:]-box[...,0:-1])),abs(numpy.amax(linarr[...,1:]-box[...,0:-1])))
				if (diff1 >= ROUGHNESS_THRESHOLD):
					box = box[::-1,...]
					leftupper = box[0,0]
					leftlower = box[x2-x1,0]
					rightupper = box[0,y2-y1]
					rightlower = box[x2-x1,y2-y1]
					xdim = box.shape[0]-1
					ydim = box.shape[1]-1
					if (leftupper > leftlower+rightupper-rightlower):
						linarr = numpy.array([[max(xx*(leftlower-leftupper)/xdim+yy*(rightupper-leftupper)/ydim+leftupper,
							  (xdim-xx)*(-rightlower+rightupper)/xdim+(ydim-yy)*(-rightlower+leftlower)/ydim+rightlower)
							       for yy in range(ydim+1)] for xx in range(xdim+1)])
					else:
						linarr = numpy.array([[min(xx*(leftlower-leftupper)/xdim+yy*(rightupper-leftupper)/ydim+leftupper,
							  (xdim-xx)*(-rightlower+rightupper)/xdim+(ydim-yy)*(-rightlower+leftlower)/ydim+rightlower)
						       for yy in range(ydim+1)] for xx in range(xdim+1)])
					diff1 = max(abs(numpy.amin(linarr-box)),abs(numpy.amax(linarr-box)));
					if (diff1 >= ROUGHNESS_THRESHOLD):
						diff1 = max(abs(numpy.amin(linarr[1:,...]-box[0:-1,...])),abs(numpy.amax(linarr[1:,...]-box[0:-1,...])))
					if (diff1 >= ROUGHNESS_THRESHOLD):
						diff1 = max(abs(numpy.amin(linarr[0:-1,...]-box[1:,...])),abs(numpy.amax(linarr[0:-1,...]-box[1:,...])))
					if (diff1 >= ROUGHNESS_THRESHOLD):
						diff1 = max(abs(numpy.amin(linarr[...,0:-1]-box[...,1:])),abs(numpy.amax(linarr[...,0:-1]-box[...,1:])))
					if (diff1 >= ROUGHNESS_THRESHOLD):
						diff1 = max(abs(numpy.amin(linarr[...,1:]-box[...,0:-1])),abs(numpy.amax(linarr[...,1:]-box[...,0:-1])))
			else:
				diff1 = numpy.amax(box)-numpy.amin(box);
			if (diff1 >= ROUGHNESS_THRESHOLD):
				split(x, x1+numpy.floor((x2-x1)/2), y1+numpy.floor((y2-y1)/2))
			else:
				x=x+1
		else:
			x=x+1

	print('done refining',len(rectList))
	pointsList = list(pointsSet)
	print('number of points:',len(pointsList))
	transformarray = numpy.zeros((len(pointsList),9),numpy.float32)
	pointsdict = dict()
	normalsdict = dict();
	for x in range(len(pointsList)):
		x1=pointsList[x][0]
		y1=pointsList[x][1]
		h1=arr[x1,y1]
		t=transform(x1,y1,h1)
		transformarray[x]=[t[0],t[1],t[2],(h1-hmin)/(hmax-hmin), 1-(h1-hmin)/(hmax-hmin),0,0,0,0]
		pointsdict[pointsList[x]] = x
		normalsdict[pointsList[x]] = numpy.zeros(3);
	print('done transforming')	
	sortByX = sorted(pointsList, key=lambda point: point[1]+100000*point[0]) 
	sortByY = sorted(pointsList, key=lambda point: point[0]+100000*point[1])
	print('sorted')
	xdict = dict()
	ydict = dict()
	for x in range(len(sortByX)):
		xdict[sortByX[x]] = x
		ydict[sortByY[x]] = x
	print('beginning triangle list')

	global edgedict
	edgedict = dict()
	triangleList = []
	for x in range(len(rectList)):
		x1 = rectList[x][0][0]
		y1 = rectList[x][0][1]
		x2 = rectList[x][1][0]
		y2 = rectList[x][1][1]
		flipped = 0
		if ((x2-x1 > 1) and (y2-y1 > 1)):
			cx = numpy.round((x2+x1)/2)
			cy = numpy.round((y2+y1)/2)
			leftupper = arr[x1,y1]
			leftlower = arr[x2,y1]
			rightupper = arr[x1,y2]
			rightlower = arr[x2,y2]
			diff1 = abs(arr[cx,cy]-(leftlower+rightupper)/2);
			diff2 = abs(arr[cx,cy]-(leftupper+rightlower)/2);
			if (diff2 < diff1):
				flipped = 1
		i1 = xdict[(x1,y1)]
		i2 = xdict[(x1,y2)]
		i3 = xdict[(x2,y1)]
		i4 = xdict[(x2,y2)]
		i5 = ydict[(x1,y1)]
		i6 = ydict[(x2,y1)]
		i7 = ydict[(x1,y2)]
		i8 = ydict[(x2,y2)]
		upper = sortByY[i5:(i6+1)]
		lower = sortByY[i7:(i8+1)]
		left = sortByX[i1:(i2+1)]
		right = sortByX[i3:(i4+1)]
		if (flipped == 1):
			left = sortByX[i3:(i4+1)]
			right = sortByX[i1:(i2+1)]
			left = [[x1,left[i][1]] for i in range(len(left))]
			right = [[x2,right[i][1]] for i in range(len(right))]
			upper = [[x2-upper[len(upper)-i-1][0]+x1,upper[len(upper)-i-1][1]] for i in range(len(upper))]
			lower = [[x2-lower[len(lower)-i-1][0]+x1,lower[len(lower)-i-1][1]] for i in range(len(lower))]
		xa = x1 
		ya = y1
		i = 1
		j = 1
		while ((i<len(left)) and (j<len(upper))):
			xleft = left[i][0] 
			yleft = left[i][1]
			xupper = upper[j][0]
			yupper = upper[j][1]
			if (flipped == 0):
				p1 = pointsdict[(xa,ya)]
				p2 = pointsdict[(xleft,yleft)]
				p3 = pointsdict[(xupper,yupper)]
			else:
				p1 = pointsdict[(x2-xa+x1,ya)]
				p2 = pointsdict[(x2-xleft+x1,yleft)]
				p3 = pointsdict[(x2-xupper+x1,yupper)]
			vectora = transformarray[p1][0:3]
			vectorright = transformarray[p2][0:3]
			vectorleft = transformarray[p3][0:3]
			cross=unitCross(vectorright[0]-vectora[0],vectorright[1]-vectora[1],
					vectorright[2]-vectora[2], -vectorleft[0]+vectorright[0],
					-vectorleft[1]+vectorright[1],-vectorleft[2]+vectorright[2])
			if (flipped == 0):
				normalsdict[(xa,ya)]=normalsdict[(xa,ya)]+cross
				normalsdict[(xleft,yleft)]=normalsdict[(xleft,yleft)]+cross
				normalsdict[(xupper,yupper)]=normalsdict[(xupper,yupper)]+cross
				edge1 = (p2,p3)
				edge2 = (p3,p1)
				edge3 = (p1,p2)
				edgedict[edge1] = p1
				edgedict[edge2] = p2
				edgedict[edge3] = p3
				triangleList.append(p1)
				triangleList.append(p2)
				triangleList.append(p3)
			else:
				normalsdict[(x2-xa+x1,ya)]=normalsdict[(x2-xa+x1,ya)]-cross
				normalsdict[(x2-xleft+x1,yleft)]=normalsdict[(x2-xleft+x1,yleft)]-cross
				normalsdict[(x2-xupper+x1,yupper)]=normalsdict[(x2-xupper+x1,yupper)]-cross
				edge1 = (p1,p2)
				edge2 = (p2,p3)
				edge3 = (p3,p1)
				edgedict[edge1] = p3
				edgedict[edge2] = p1
				edgedict[edge3] = p2
				triangleList.append(p3)
				triangleList.append(p1)
				triangleList.append(p2)
			if ((yleft-y1 >= xupper-x1) and j<(len(upper)-1)):
				j = j+1
				xa = xupper
				ya = yupper
			elif (i<len(left)-1):
				i = i+1
				xa = xleft
				ya = yleft
			else:
				j = j+1
				xa = xupper
				ya = yupper
		i = len(right)-2; j = len(lower)-2
		xa = x2
		ya = y2
		while (i>=0) and (j>=0):
			xright = right[i][0] 
			yright = right[i][1]
			xlower = lower[j][0]
			ylower = lower[j][1]
			if (flipped == 0):
				p1 = pointsdict[(xa,ya)]
				p2 = pointsdict[(xright,yright)]
				p3 = pointsdict[(xlower,ylower)]
			else:
				p1 = pointsdict[(x2-xa+x1,ya)]
				p2 = pointsdict[(x2-xright+x1,yright)]
				p3 = pointsdict[(x2-xlower+x1,ylower)]
			vectora = transformarray[p1][0:3]
			vectorright = transformarray[p2][0:3]
			vectorleft = transformarray[p3][0:3]
			cross=unitCross(vectorright[0]-vectora[0],vectorright[1]-vectora[1],
					vectorright[2]-vectora[2], -vectorleft[0]+vectorright[0],
					-vectorleft[1]+vectorright[1],-vectorleft[2]+vectorright[2])
			if (flipped == 0):
				normalsdict[(xa,ya)]=normalsdict[(xa,ya)]+cross
				normalsdict[(xright,yright)]=normalsdict[(xright,yright)]+cross
				normalsdict[(xlower,ylower)]=normalsdict[(xlower,ylower)]+cross
				edge1 = (p1,p2)
				edge2 = (p2,p3)
				edge3 = (p3,p1)
				edgedict[edge1] = p3
				edgedict[edge2] = p1
				edgedict[edge3] = p2
				triangleList.append(p3)
				triangleList.append(p1)
				triangleList.append(p2)
			else:
				normalsdict[(x2-xa+x1,ya)]=normalsdict[(x2-xa+x1,ya)]-cross
				normalsdict[(x2-xright+x1,yright)]=normalsdict[(x2-xright+x1,yright)]-cross
				normalsdict[(x2-xlower+x1,ylower)]=normalsdict[(x2-xlower+x1,ylower)]-cross
				edge1 = (p2,p3)
				edge2 = (p3,p1)
				edge3 = (p1,p2)
				edgedict[edge1] = p1
				edgedict[edge2] = p2
				edgedict[edge3] = p3
				triangleList.append(p1)
				triangleList.append(p2)
				triangleList.append(p3)
			if ((y2-yright >= x2-xlower) and j > 0):
				j = j-1
				xa = xlower
				ya = ylower
			elif (i > 0):
				i = i-1
				xa = xright
				ya = yright
			else:
				j = j-1
				xa = xlower
				ya = ylower
		if x%1000 == 0:
			print(x)

	for x in range(len(pointsList)):
		norm = numpy.sqrt(normalsdict[pointsList[x]][0]**2+normalsdict[pointsList[x]][1]**2+normalsdict[pointsList[x]][2]**2)
		if (norm > 0):
			transformarray[x][6] = normalsdict[pointsList[x]][0]/norm
			transformarray[x][7] = normalsdict[pointsList[x]][1]/norm
			transformarray[x][8] = normalsdict[pointsList[x]][2]/norm

	adjacencyList = []
	for x in range(len(triangleList)/3):
		p1 = triangleList[3*x]
		p2 = triangleList[3*x+1]
		p3 = triangleList[3*x+2]
		p4 = edgedict.get((p3,p2))
		p5 = edgedict.get((p1,p3))
		p6 = edgedict.get((p2,p1))
		if (p4 == None):
			p4 = p2
		if (p5 == None):
			p5 = p1
		if (p6 == None):
			p6 = p1
		adjacencyList.extend([p1,p2,p3,p4,p5,p6])
	
	if (WRITEOUT==1):
		north = transform(0,YCENTER,arr[0,YCENTER]);
		nvec = numpy.array([north[0],north[1],north[2]]);
		nvec2 = numpy.dot(nvec,cameraview);
		nangle = math.pi/2-math.atan2(nvec2[2],nvec2[0]);
		rotmatrix = numpy.array([[math.cos(nangle), 0, math.sin(nangle)],[0,1,0],[-math.sin(nangle),0,math.cos(nangle)]]);
		fmatrix = numpy.dot(cameraview,rotmatrix);
		normals = transformarray[...,6:9]
		normals = numpy.dot(normals,fmatrix);
		coords = transformarray[...,0:3]
		coords = numpy.dot(coords,fmatrix)
		colors = transformarray[...,4:5]
		coordslist = numpy.array([coords[i,j] for i in range(len(pointsList)) for j in range(3)])
		normalslist = numpy.array([normals[i,j] for i in range(len(pointsList)) for j in range(3)])
		colors = colors.transpose()
		colors = colors[0].tolist()
		tlist = [fmatrix[0][0], fmatrix[0][1], fmatrix[0][2], 0,
			 fmatrix[1][0], fmatrix[1][1], fmatrix[1][2], 0,
			 fmatrix[2][0], fmatrix[2][1], fmatrix[2][2], 0,
			 0, 0, 0, 1]
		stlist = [1/SCALEFACTOR, 0, 0, 0,
			  0, 1/SCALEFACTOR, 0, 0,
			  0, 0, 1/SCALEFACTOR, 0,
			  -xc/SCALEFACTOR, -yc/SCALEFACTOR, -zc/SCALEFACTOR, 1]
		with open('dataList.js', 'w') as f:
			f.write('/* coordinates: X0= '+str(X0)+' X1= '+str(X1)+' Y0= '+str(Y0)+' Y1= '+str(Y1)+' */ ');
			f.write('var stmatrix =');
			json.dump(stlist,f);
			f.write('; var fmatrix =');
			json.dump(tlist,f);
			f.write('; var triangleList =')
			json.dump(triangleList,f)
			json.encoder.FLOAT_REPR = lambda o: format(o,'.3f')
			f.write('; var coords =');
			json.dump(coordslist.tolist(),f)
			f.write('; var colors =');
			json.dump(colors,f)
			f.write('; var normals =');
			json.dump(normalslist.tolist(),f)
			f.write(';')

	print(len(triangleList))
	print(len(adjacencyList))
	global numTr
	numTr = len(triangleList)
	global theVBO
	theVBO = vbo.VBO(transformarray, target=GL_ARRAY_BUFFER)
	global indexBuffer
	indexarray = numpy.array(triangleList, numpy.uint32)
	indexBuffer = vbo.VBO(indexarray, target=GL_ELEMENT_ARRAY_BUFFER)
	global adjIndexBuffer
	adjindexarray = numpy.array(adjacencyList, numpy.uint32)
	adjIndexBuffer = vbo.VBO(adjindexarray, target=GL_ELEMENT_ARRAY_BUFFER)

def unitCross(x1, y1, z1, x2, y2, z2):
	cross = numpy.zeros(3);
	cross[0] = y1*z2-y2*z1;
	cross[1] = -x1*z2+x2*z1;
	cross[2] = x1*y2-y1*x2;
	norm = numpy.sqrt(cross[0]**2+cross[1]**2+cross[2]**2);
	if (norm != 0):
		return cross/norm;
	else:
		return cross;

def getAngle(x1,y1,x2,y2):
	n1 = math.sqrt(x1**2+y1**2)
	n2 = math.sqrt(x2**2+y2**2)
	cr = abs(x1*y2-x2*y1)
	a = math.asin(cr/n1/n2)
	if (x1*x2+y1*y2 < 0):
		return math.pi-a
	else:
		return a

def transform(x1, y1, h):
	lon = math.radians(LONMIN+float(y1)/8100);
	lat = math.radians(LATMAX-float(x1)/8100);
	n = a/numpy.sqrt(1-e2*(math.sin(lat))**2)
	fx = (n+h)*math.cos(lat)*math.cos(lon)-xc
	fy = (n+h)*math.cos(lat)*math.sin(lon)-yc
	fz = (n*(1-e2)+h)*math.sin(lat)-zc
	return [fx/SCALEFACTOR,fy/SCALEFACTOR,fz/SCALEFACTOR]

# The function called when our window is resized (which shouldn't happen if you enable fullscreen, below)
def ReSizeGLScene(Width, Height):
    if Height == 0:						# Prevent A Divide By Zero If The Window Is Too Small 
	    Height = 1

    global W
    global H
    global viewangle
    W = Width
    H = Height
    glViewport(0, 0, Width, Height)		# Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(viewangle, float(Width)/float(Height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

# The main drawing function. 
def DrawGLScene():
	global ridgemode, xlevel, ylevel, zlevel
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	# Clear The Screen And The Depth Buffer

	glLoadIdentity();					# Reset The View
	glRotatef(phi,1.0,0.0,0.0);
	glRotatef(theta,0.0,1.0,0.0);				# Rotate The Pyramid On It's Y Axis
	glTranslatef(-xlevel,-zlevel,-ylevel,0);
	glRotatef(90,-1.0,0,0);
	glRotatef(math.degrees(-math.pi/2+math.atan2(zc, numpy.sqrt(xc**2+yc**2))), -yc, xc, 0);
	glUseProgram(shaders);
	try:
		theVBO.bind()
		indexBuffer.bind()
		try:
			glEnableClientState(GL_VERTEX_ARRAY)
			glEnableClientState(GL_COLOR_ARRAY)
			glEnableClientState(GL_NORMAL_ARRAY)
			glVertexPointer(3,GL_FLOAT,36,theVBO)
			glColorPointer(3,GL_FLOAT,36,theVBO+12)
			glNormalPointer(GL_FLOAT,36,theVBO+24)
			light = glGetUniformLocation(shaders, 'light');
			glUniform3f(light,LIGHT[0],LIGHT[1],LIGHT[2])
			glDrawElements(GL_TRIANGLES,numTr,GL_UNSIGNED_INT,None)
		finally:
			indexBuffer.unbind()
			theVBO.unbind()
		if (ridgemode==1 and GEOSHADERS==1):
			glUseProgram(ridgeprogram)
			adjIndexBuffer.bind()
			theVBO.bind()
			try:
				glEnableClientState(GL_VERTEX_ARRAY)
				glEnableClientState(GL_COLOR_ARRAY)
				glVertexPointer(3,GL_FLOAT,36,theVBO)
				glColorPointer(3,GL_FLOAT,36,theVBO+12)
				glDrawElements(GL_TRIANGLES_ADJACENCY,2*numTr,GL_UNSIGNED_INT,None)
			finally:
				adjIndexBuffer.unbind()
				theVBO.unbind()
	finally:
		glUseProgram(0)

	#  since this is double buffered, swap the buffers to display what just got drawn. 
	glutSwapBuffers()

# The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)  
def keyPressed(*args):
	global zlevel, viewangle, W, H, ridgemode, xlevel, ylevel, theta
	# If escape is pressed, kill everything.
	if args[0] == ESCAPE:
		sys.exit()
	elif args[0] == PLUS:
		zlevel = zlevel+0.02
	elif args[0] == MINUS:
		zlevel = zlevel-0.02
	elif args[0] == A:
		if (viewangle < 45.0):
			viewangle = viewangle+0.5;
			glMatrixMode(GL_PROJECTION)
			glLoadIdentity()
			gluPerspective(viewangle, float(W)/float(H), 0.1, 100.0)
			glMatrixMode(GL_MODELVIEW)
	elif args[0] == Z:
		if (viewangle > 10.0):
			viewangle = viewangle-0.5;
			glMatrixMode(GL_PROJECTION)
			glLoadIdentity()
			gluPerspective(viewangle, float(W)/float(H), 0.1, 100.0)
			glMatrixMode(GL_MODELVIEW)
	elif args[0] == R:
		if (ridgemode == 0):
			ridgemode = 1;
		elif (ridgemode == 1):
			ridgemode = 0;
	elif args[0] == J:
		xlevel = xlevel-0.05*math.cos(math.radians(theta))
		ylevel = ylevel-0.05*math.sin(math.radians(theta))
	elif args[0] == K:
		xlevel = xlevel+0.05*math.cos(math.radians(theta+90))
		ylevel = ylevel+0.05*math.sin(math.radians(theta+90))
	elif args[0] == L:
		xlevel = xlevel+0.05*math.cos(math.radians(theta))
		ylevel = ylevel+0.05*math.sin(math.radians(theta))
	elif args[0] == I:
		xlevel = xlevel-0.05*math.cos(math.radians(theta+90))
		ylevel = ylevel-0.05*math.sin(math.radians(theta+90))
	
	
def specialKeyPressed(*args):
	global theta, phi
	if args[0] == GLUT_KEY_LEFT:
		theta = theta-0.4
	elif args[0] == GLUT_KEY_RIGHT:
		theta = theta+0.4
	elif args[0] == GLUT_KEY_UP:
		phi = phi-0.4
	elif args[0] == GLUT_KEY_DOWN:
		phi = phi+0.4
	
def main():
	global window
	glutInit(sys.argv)

	# Select type of Display mode:   
	#  Double buffer 
	#  RGBA color
	# Alpha components supported 
	# Depth buffer
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
	
	# get a 640 x 480 window 
	glutInitWindowSize(640, 480)
	
	# the window starts at the upper left corner of the screen 
	glutInitWindowPosition(0, 0)
	
	# Okay, like the C version we retain the window id to use when closing, but for those of you new
	# to Python (like myself), remember this assignment would make the variable local and not global
	# if it weren't for the global declaration at the start of main.
	window = glutCreateWindow("Terrain")

	# Register the drawing function with glut, BUT in Python land, at least using PyOpenGL, we need to
	# set the function pointer and invoke a function to actually register the callback, otherwise it
	# would be very much like the C version of the code.	
	glutDisplayFunc(DrawGLScene)
	
	# Uncomment this line to get full screen.
	#glutFullScreen()

	# When we are doing nothing, redraw the scene.
	glutIdleFunc(DrawGLScene)
	
	# Register the function called when our window is resized.
	glutReshapeFunc(ReSizeGLScene)
	
	# Register the function called when the keyboard is pressed.  
	glutKeyboardFunc(keyPressed)

	glutSpecialFunc(specialKeyPressed)

	# Initialize our window. 
	InitGL(640, 480)

	# Start Event Processing Engine	
	glutMainLoop()

# Print message to console, and kick off the main to get it rolling.
print "Hit ESC key to quit."
main()
	
