import maya.cmds as mc
import sys
sys.path.append(u'C:/Users/qenops/Dropbox/code/python')
sys.path.append(u'C:/Python27/Lib/site-packages/')
import dGraph as dg
import numpy as np
import math

def getPOSInode(surf):
    # Will create a maya posi node if one doesn't exist, or return the existing one it if it already does
    try:
        posi = mc.listConnections('%s.worldSpace'%surf,t='pointOnSurfaceInfo')[0]
    except TypeError:
        posi = mc.createNode('pointOnSurfaceInfo')
        mc.connectAttr('%s.worldSpace[0]'%surf, '%s.inputSurface'%posi)
    return posi
def getCPOSnode(surf):
    # Will create a maya posi node if one doesn't exist, or return the existing one it if it already does
    try:
        cpos = mc.listConnections('%s.worldSpace'%surf,t='closestPointOnSurface')[0]
    except TypeError:
        cpos = mc.createNode('closestPointOnSurface')
        mc.connectAttr('%s.worldSpace[0]'%surf, '%s.inputSurface'%cpos)
    return cpos
def getReflectedRay(mirror, eyePoint, param):
    # Given a ray and a surface, finds the reflected ray
    posi = getPOSInode(mirror)
    mc.setAttr('%s.parameterU'%posi, param[0])
    mc.setAttr('%s.parameterV'%posi, param[1])
    target = np.array(mc.getAttr('%s.position'%posi)[0])
    myRay = dg.Ray(eyePoint,target-eyePoint)
    normal = np.array(mc.getAttr('%s.normal'%posi)[0])
    normal = normal/np.linalg.norm(normal)
    reflectVect = myRay.vector-2*(np.dot(myRay.vector,normal))*normal
    reflectRay = dg.Ray(target,reflectVect)
    return reflectRay, myRay
def getReflectionPoint(mirror, pntA, pntB, param=np.array((.5,.5)), step=.005, iter=20):
    plus = np.array((0.,0.))
    minus = np.array((0.,0.))
    # Finds the best surface point which will reflect pntA to pntB using stochastic gradient descent
    reflectRay, myRay = getReflectedRay(mirror, pntA, param)
    dist = reflectRay.distanceToPoint(pntB)
    reflectRay, myRay = getReflectedRay(mirror, pntA, param+np.array((step,0)))
    distU = reflectRay.distanceToPoint(pntB)
    reflectRay, myRay = getReflectedRay(mirror, pntA, param+np.array((0,step)))
    distV = reflectRay.distanceToPoint(pntB)
    reflectRay, myRay = getReflectedRay(mirror, pntA, param+np.array((step,step)))
    distUV = reflectRay.distanceToPoint(pntB)
    # Find the intercept
    plus[0]=param[0]+dist*(-step/(distU-dist))
    plus[1]=param[1]+dist*(-step/(distV-dist))
    minus[0]=param[0]-dist*(-step/(distU-dist))
    minus[1]=param[1]-dist*(-step/(distV-dist))
    reflectRay, myRay = getReflectedRay(mirror, pntA, plus)
    dist = reflectRay.distanceToPoint(pntB)
    reflectRay, myRay = getReflectedRay(mirror, pntA, minus)
    dist = reflectRay.distanceToPoint(pntB)
    step = dist*(step/(distU-dist))/10
    return 
def getReflectionPoint2(mirror, pntA, pntB, param=np.array((.5,.5)), step=.01, precision=5):
    # just sample the points for this precision level, chooses the best one, then goes deeper
    for x in range(precision):
        store = np.zeros((10,10))
        Uval=-step*5
        Vval=-step*5
        for U in range(10):
            for V in range(10):
                reflectRay, myRay = getReflectedRay(mirror, pntA, param+np.array((Uval+U*step,Vval+V*step)))
                store[U,V] = reflectRay.distanceToPoint(pntB)
        idx = np.argmin(store)
        U = idx/10
        V = idx%10
        param+=np.array((Uval+U*step,Vval+V*step))
        step/=10
    return reflectRay.point #,param
def getVergencePoint(listOfRays):
    # Finds the nearest point of vergence for all the given rays
    sum1 = np.zeros((3,3))
    sum2 = np.zeros((3,1))
    for ray in listOfRays:
        first = np.eye(3)-ray.vector[np.newaxis].T*ray.vector
        second = np.dot(first,ray.point[np.newaxis].T)
        sum1 += first
        sum2 += second
    return np.dot(np.linalg.inv(sum1),sum2).T[0]
def reflectImagePlane(mirror, imagePlane, listOfViews, uRange=np.arange(.1,1,.1), vRange=np.arange(.1,1,.1), precision=4):
    # Create an object representing the virtual image reflected in a mirror from the viewpoints given
    points = np.zeros((len(uRange),len(vRange),3))
    for uidx, u in enumerate(uRange):
        print u
        for vidx, v in enumerate(vRange):
            param = np.array((u,v)) 
            reflectRay, origRay = getReflectedRay(mirror, listOfViews[0], param)
            intersection = screen.intersection(reflectRay)
            rayBundle = []
            if intersection != []:
                rayBundle.append(origRay)
                screenPoint = intersection[0]['point']
                #mc.curve(d=1, p=[origRay.point.tolist(), reflectRay.point.tolist()])
                #mc.curve(d=1, p=[reflectRay.point.tolist(), screenPoint.tolist()])
                for pntA in listOfViews[1:]:
                    reflectPnt = getReflectionPoint2(mirror, pntA, screenPoint, param, precision=precision)
                    rayBundle.append(dg.Ray(pntA,reflectPnt-pntA))
                    #mc.curve(d=1, p=[pntA.tolist(), reflectPnt.tolist()])
                    #mc.curve(d=1, p=[reflectPnt.tolist(), screenPoint.tolist()])
                vergencePnt = getVergencePoint(rayBundle)
                points[uidx,vidx] = vergencePnt
    return points
def drawPoly(points):
    # create a polygonal surface visualization of the given set of points
    sx, sy, sz = points.shape
    if sz != 3:
        raise IndexError('Point locations must be 3 dimensional')
    vi = mc.polyPlane(sx=sx-1, sy=sy-1,w=0.1,h=0.1)[0]
    mc.setAttr('%s.vtx[0:%d]'%(vi,sx*sy-1), *points.flatten().tolist())  # This won't be exact, since the inMesh has cached positions
    return vi
def keyPoly(points, vi):
    # keyframe a polygonal surface visualization of the given set of points at this frame
    sx, sy, sz = points.shape
    if sz != 3:
        raise IndexError('Point locations must be 3 dimensional')
    mc.setAttr('%s.vtx[0:%d]'%(vi,sx*sy-1), *points.flatten().tolist())  # This won't be exact, since the inMesh has cached positions
    mc.setKeyframe('%s.vtx[0:%d]'%(vi,sx*sy-1))
    return vi
def fftConvo(data, kernel, mode='extend', tol=1e-15):
    ''' Use fft to compute the 2d convolution of a matrix given a kernel '''
    # pad the arrays data=points[:,:,0] kernel=k
    tshp = data.shape[:2]
    kshp = kernel.shape
    e1 = np.zeros(np.array(tshp) + np.array(kshp) - np.array((1,1)))
    e1[:tshp[0],:tshp[1]] = data
    # extend the borders - these will need to be reworked for kernels larger than 3x3
    if mode=='repeat':
        e1[tshp[0]:e1.shape[0]-kshp[0]/2,:tshp[1]] = data[-1,:]
        e1[e1.shape[0]-kshp[0]/2:,:tshp[1]] = data[0,:]
        e1[:,tshp[1]:e1.shape[1]-kshp[1]/2] = e1[:,tshp[1]-1][np.newaxis].T
        e1[:,e1.shape[1]-kshp[1]/2:] = e1[:,0][np.newaxis].T
    if mode=='extend':
        e1[tshp[0]:e1.shape[0]-kshp[0]/2,:tshp[1]] = 2*data[-1,:]-data[-2,:]
        e1[e1.shape[0]-kshp[0]/2:,:tshp[1]] = 2*data[0,:]-data[1,:]
        e1[:,tshp[1]:e1.shape[1]-kshp[1]/2] = (2*e1[:,tshp[1]-1]-e1[:,tshp[1]-2])[np.newaxis].T
        e1[:,e1.shape[1]-kshp[1]/2:] = (2*e1[:,0]-e1[:,1])[np.newaxis].T
    if mode=='extend2':
        e1[tshp[0]:e1.shape[0]-kshp[0]/2,:tshp[1]] = 2*(data[-2,:]-data[-3,:])+data[-2,:]
        e1[e1.shape[0]-kshp[0]/2:,:tshp[1]] = 2*(data[1,:]-data[2,:])+data[1,:]
        e1[:,tshp[1]:e1.shape[1]-kshp[1]/2] = (2*(e1[:,tshp[1]-2]-e1[:,tshp[1]-3])+e1[:,tshp[1]-2])[np.newaxis].T
        e1[:,e1.shape[1]-kshp[1]/2:] = (2*(e1[:,1]-e1[:,2])+e1[:,1])[np.newaxis].T
    # place our kernel appropriately
    e2 = np.zeros_like(e1)
    e2[:math.ceil(kshp[0]/2.),:math.ceil(kshp[1]/2.)] = kernel[kshp[0]/2:,kshp[1]/2:]
    e2[e2.shape[0]-kshp[0]/2:,e2.shape[1]-kshp[1]/2:] = kernel[:kshp[0]/2,:kshp[1]/2]
    e2[:math.ceil(kshp[0]/2.),e2.shape[1]-kshp[1]/2:] = kernel[kshp[0]/2:,:kshp[1]/2]
    e2[e2.shape[0]-kshp[0]/2:,:math.ceil(kshp[1]/2.)] = kernel[:kshp[0]/2,kshp[1]/2:]
    # use fft to compute the convolution
    f1 = np.fft.fft2(e1)
    f2 = np.fft.fft2(e2)
    f3 = f1*f2
    r = np.real(np.fft.ifft2(f3))
    r[abs(r) < tol] = 0.
    return r[:tshp[0],:tshp[1]]
def calcSmoothness(points):
    # Approximate the Geometric Laplacian to calculate the surface smoothness of each vertex
    neighbors = np.zeros_like(points)
    k=np.array(([.0888,.1612,.0888],[.1612,0.,.1612],[.0888,.1612,.0888]))
    for i in range(3):
        neighbors[:,:,i] = fftConvo(points[:,:,i], k)
    return np.sum((points-neighbors)**2,axis=2)
def calcDistance(points, eye=np.array((-32.5,0.,80.))):
    x = np.full(points.shape,eye)
    return -np.sum((x - points)**2,axis=2)
def mapInfluences(mirror, screen, pupilPoints, uRange, vRange):
    uRange=np.arange(.1,1,.1)
    vRange=np.arange(.1,1,.1)
    points = reflectImagePlane(mirror, screen, pupilPoints, uRange=uRange, vRange=vRange, precision=3)   
    baseline = calcSmoothness(points)
    max = np.zeros((len(uRange),len(vRange)))
    idx = np.zeros((len(uRange),len(vRange),2),dtype=int)
    for u in range(mc.getAttr('%s.spansU'%mirror)+mc.getAttr('%s.degreeU'%mirror)+1):
        for v in range(mc.getAttr('%s.spansV'%mirror)+mc.getAttr('%s.degreeV'%mirror)+1):
            print '\n%s,%s'%(u,v)
            mc.move(0, -1, 0, '%s.cv[%s][%s]'%(mirror,u,v), r=1, os=1, wd=1)
            points = reflectImagePlane(mirror, screen, pupilPoints, uRange=uRange, vRange=vRange, precision=3)  
            new = calcSmoothness(points)
            mc.move(0, 1, 0, '%s.cv[%s][%s]'%(mirror,u,v), r=1, os=1, wd=1)
            diff = abs((baseline - new)/baseline)
            max = np.maximum(max, diff)
            idx[max == diff] = np.array((u,v),dtype=int)
    return idx
def optimizeMirrorShape(mirror, screen, pupilPoints, remap, iterations=5, uRange=np.arange(.1,1,.1), vRange=np.arange(.1,1,.1), costFunction=calcSmoothness, factor=1):
    best = mc.group(em=1, n='best')
    bMirror = mc.duplicate(mirror,n='best%s'%mirror)[0]
    mc.parent(bMirror,best)
    points = reflectImagePlane(mirror, screen, pupilPoints, uRange=uRange, vRange=vRange, precision=3)
    cost = costFunction(points)
    shape = cost.shape
    skip = -1
    for n in range(iterations):
        print n
        # find the cv to optimize
        i = np.argsort(cost.flatten())[skip]
        x,y = np.unravel_index(i,shape)
        u, v = remap[x,y]
        mc.move(0, factor, 0, '%s.cv[%s][%s]'%(bMirror,u,v), r=1, os=1, wd=1)
        pPoints = reflectImagePlane(bMirror, screen, pupilPoints, uRange=uRange, vRange=vRange, precision=3)  
        pos = costFunction(pPoints)
        if (np.sum(pos) < np.sum(cost)): #  pos[x,y] < cost[x,y]
            points = pPoints
            cost = pos
            continue
        mc.move(0, -factor, 0, '%s.cv[%s][%s]'%(bMirror,u,v), r=1, os=1, wd=1)
        mc.move(0, -factor, 0, '%s.cv[%s][%s]'%(bMirror,u,v), r=1, os=1, wd=1)
        nPoints = reflectImagePlane(bMirror, screen, pupilPoints, uRange=uRange, vRange=vRange, precision=3)  
        neg = costFunction(nPoints)
        if (np.sum(pos) < np.sum(cost)): #  neg[x,y] < cost[x,y]
            points = nPoints
            cost = neg
            #factor=-factor
            continue
        mc.move(0, factor, 0, '%s.cv[%s][%s]'%(bMirror,u,v), r=1, os=1, wd=1)
        #if random() < (skip/2. + 1.5):
        #    skip -= 1
        #else:
        #    skip += 1
        skip -= 1
        print ('No improvement, skipping to %s'%skip)
    return points
def radiallySort(shape):
    subtract = np.array(shape)/2. - np.array((.5,.5))
    values = np.indices(shape)
    values[0] = values[0] - subtract[0]
    values[1] = values[1] - subtract[1]
    return np.argsort((values[0]**2+values[1]**2).flatten())
def optimizeRankedMirrorShape(mirror, screen, pupilPoints, remap, rank, iterations=5, uRange=np.arange(.1,1,.1), vRange=np.arange(.1,1,.1), costFunction=calcSmoothness, factor=1):
    best = mc.group(em=1, n='best')
    bMirror = mc.duplicate(mirror,n='best%s'%mirror)[0]
    mc.parent(bMirror,best)
    points = reflectImagePlane(mirror, screen, pupilPoints, uRange=uRange, vRange=vRange, precision=3)
    cost = costFunction(points)
    shape = cost.shape
    skip = 0
    for n in range(iterations):
        print n
        # find the cv to optimize
        i = rank[skip]
        x,y = np.unravel_index(i,shape)
        u, v = remap[x,y]
        mc.move(0, factor, 0, '%s.cv[%s][%s]'%(bMirror,u,v), r=1, os=1, wd=1)
        pPoints = reflectImagePlane(bMirror, screen, pupilPoints, uRange=uRange, vRange=vRange, precision=3)  
        pos = costFunction(pPoints)
        if (np.sum(pos) < np.sum(cost)): #  pos[x,y] < cost[x,y]
            points = pPoints
            cost = pos
            continue
        mc.move(0, -factor, 0, '%s.cv[%s][%s]'%(bMirror,u,v), r=1, os=1, wd=1)
        mc.move(0, -factor, 0, '%s.cv[%s][%s]'%(bMirror,u,v), r=1, os=1, wd=1)
        nPoints = reflectImagePlane(bMirror, screen, pupilPoints, uRange=uRange, vRange=vRange, precision=3)  
        neg = costFunction(nPoints)
        if (np.sum(pos) < np.sum(cost)): #  neg[x,y] < cost[x,y]
            points = nPoints
            cost = neg
            #factor=-factor
            continue
        mc.move(0, factor, 0, '%s.cv[%s][%s]'%(bMirror,u,v), r=1, os=1, wd=1)
        #if random() < (skip/2. + 1.5):
        #    skip -= 1
        #else:
        #    skip += 1
        skip += 1
        print ('No improvement, skipping to %s'%skip)
    return points
def raySurfaceIntersect(surf, ray, tol=1e-12):
    '''Find the surface point that intersects the ray '''
    cpos = getCPOSnode(surf)
    spnt = np.array(mc.xform(surf, q=1,t=1,ws=1))
    for i in range(100):
        rpnt = ray.projectPointOnRay(spnt)
        mc.setAttr('%s.inPosition'%cpos,*rpnt)
        spnt = np.array(mc.getAttr('%s.position'%cpos)[0])
        if np.sum((rpnt - spnt)**2) < tol:
            #print i
            break
    return mc.getAttr('%s.parameterU'%cpos), mc.getAttr('%s.parameterV'%cpos)
def pointSpreadCost(mirror, screen, pupilPoints, virtualImage):
    posi = getPOSInode(mirror)
    count = mc.polyEvaluate(virtualImage, v=1)
    cumulativeDistance = np.zeros(count)
    for i in range(count):
        virtualPnt = np.array(mc.xform('%s.vtx[%s]'%(virtualImage,i),q=1,ws=1,t=1))
        ray1 = dg.Ray(pupilPoints[0],virtualPnt-pupilPoints[0])
        param = raySurfaceIntersect(mirror, ray1)
        reflectRay, origRay = getReflectedRay(mirror, pupilPoints[0], param)
        intersection1 = screen.intersection(reflectRay)[0]
        cumulativeDistance[i] = 0
        for eyePnt in pupilPoints[1:]:
            ray2 = dg.Ray(eyePnt,virtualPnt-eyePnt)
            param = raySurfaceIntersect(mirror, ray2)
            reflectRay, origRay = getReflectedRay(mirror, eyePnt, param)
            intersection2 = screen.intersection(reflectRay)[0]
            cumulativeDistance[i] += np.sum((intersection1['point']-intersection2['point'])**2)
    return cumulativeDistance
def normalCost(mirror, screen, pupilPoints, virtualImage):
    posi = getPOSInode(mirror)
    count = mc.polyEvaluate(virtualImage, v=1)
    cumulativeCost = np.zeros(count)
    for i in range(count):
        virtualPnt = np.array(mc.xform('%s.vtx[%s]'%(virtualImage,i),q=1,ws=1,t=1))
        ray1 = dg.Ray(pupilPoints[0],virtualPnt-pupilPoints[0])
        param = raySurfaceIntersect(mirror, ray1)
        reflectRay, origRay = getReflectedRay(mirror, pupilPoints[0], param)
        intersection1 = screen.intersection(reflectRay)[0]
        cumulativeCost[i] = 0
        for eyePnt in pupilPoints[1:]:
            ray2 = dg.Ray(eyePnt,virtualPnt-eyePnt)
            param = raySurfaceIntersect(mirror, ray2)
            reflectRay, origRay = getReflectedRay(mirror, eyePnt, param)
            goalRay = dg.Ray(reflectRay.point,intersection1['point']-reflectRay.point)
            cumulativeCost[i] += 1-np.dot(goalRay.vector,reflectRay.vector)
    return cumulativeCost
def mapImageInfluences(mirror, screen, pupilPoints, virtualImage, costFunction=normalCost):
    baseline = costFunction(mirror, screen, pupilPoints, virtualImage)
    maximum = np.zeros_like(baseline)
    idx = np.zeros(baseline.shape+(2L,),dtype=int)
    for u in range(mc.getAttr('%s.spansU'%mirror)+mc.getAttr('%s.degreeU'%mirror)):
        for v in range(mc.getAttr('%s.spansV'%mirror)+mc.getAttr('%s.degreeV'%mirror)):
            print '\n%s,%s'%(u,v)
            mc.move(0, -1, 0, '%s.cv[%s][%s]'%(mirror,u,v), r=1, os=1, wd=1)
            new = costFunction(mirror, screen, pupilPoints, virtualImage)
            mc.move(0, 1, 0, '%s.cv[%s][%s]'%(mirror,u,v), r=1, os=1, wd=1)
            diff = abs((baseline - new)/baseline)
            maximum = np.maximum(maximum, diff)
            idx[maximum == diff] = np.array((u,v),dtype=int)
    return idx
def optimizeMirrorForImage(mirror, screen, pupilPoints, virtualImage, remap, iterations=5, costFunction=normalCost, factor=-1):
    best = mc.group(em=1, n='best')
    bMirror = mc.duplicate(mirror,n='best%s'%mirror)[0]
    mc.parent(bMirror,best)
    cost = costFunction(mirror, screen, pupilPoints, virtualImage)
    shape = cost.shape
    skip = -1
    count = 0
    for n in range(iterations):
        print n
        # find the cv to optimize
        i = np.argsort(cost)[skip]
        #x,y = np.unravel_index(i,shape)
        u, v = remap[i]
        mc.move(0, factor, 0, '%s.cv[%s][%s]'%(bMirror,u,v), r=1, os=1, wd=1)
        pos = costFunction(bMirror, screen, pupilPoints, virtualImage)
        if (np.sum(pos) < np.sum(cost)): #  pos[x,y] < cost[x,y]
            cost = pos
            count = 0
            continue
        mc.move(0, -factor, 0, '%s.cv[%s][%s]'%(bMirror,u,v), r=1, os=1, wd=1)
        mc.move(0, -factor, 0, '%s.cv[%s][%s]'%(bMirror,u,v), r=1, os=1, wd=1)
        neg = costFunction(bMirror, screen, pupilPoints, virtualImage)
        if (np.sum(pos) < np.sum(cost)): #  neg[x,y] < cost[x,y]
            cost = neg
            print "neg"
            #factor=-factor
            count = 0
            continue
        mc.move(0, factor, 0, '%s.cv[%s][%s]'%(bMirror,u,v), r=1, os=1, wd=1)
        #if random() < (skip/2. + 1.5):
        #    skip -= 1
        #else:
        #    skip += 1
        skip -= 1
        print ('No improvement, skipping to %s'%skip)
        count += 1
        if count > 10:
            factor /= 10.
            print ('New factor: %s'%factor)
            count = 0
            skip = -1
    return bMirror
def optimizeCVsForImage(mirror, screen, pupilPoints, virtualImage, costFunction=normalCost, start=-1., iterations=5, border=False):
    # work your way through each cv improving until it can't get better
    best = mc.group(em=1, n='best')
    bMirror = mc.duplicate(mirror,n='best%s'%mirror)[0]
    mc.parent(bMirror,best)
    cvU = mc.getAttr('%s.spansU'%mirror)+mc.getAttr('%s.degreeU'%mirror)
    cvV = mc.getAttr('%s.spansV'%mirror)+mc.getAttr('%s.degreeV'%mirror)
    shape = (cvU, cvV)
    order = radiallySort(shape)
    cost = costFunction(mirror, screen, pupilPoints, virtualImage)
    for factor in [start/3.**exp for exp in range(0,iterations)]: #13)]:
        print factor
        for i in order:
            u,v = np.unravel_index(i,shape)
            if not border:
                if u==0 or u==cvU-1 or v==0 or v==cvV-1:
                    continue
            print '%s,%s'%(u,v)
            update = 0
            pos = np.zeros((3))
            while (np.sum(pos) < np.sum(cost)):
                print update
                if update > 0:
                    cost = pos
                mc.move(0, factor, 0, '%s.cv[%s][%s]'%(bMirror,u,v), r=1, os=1, wd=1)
                pos = costFunction(bMirror, screen, pupilPoints, virtualImage)
                update += 1
                if update > 10:
                    break
            mc.move(0, -factor, 0, '%s.cv[%s][%s]'%(bMirror,u,v), r=1, os=1, wd=1)
            if update < 2:
                update = 0
                neg = np.zeros((3))
                while (np.sum(neg) < np.sum(cost)):
                    print update
                    if update > 0:
                        cost = neg
                    mc.move(0, -factor, 0, '%s.cv[%s][%s]'%(bMirror,u,v), r=1, os=1, wd=1)
                    neg = costFunction(bMirror, screen, pupilPoints, virtualImage)
                    update += 1
                    if update > 10:
                        break
                mc.move(0, factor, 0, '%s.cv[%s][%s]'%(bMirror,u,v), r=1, os=1, wd=1)
    return bMirror


iter = 10
s = time.clock()
for i in range(iter):
    pointSpreadCost(mirror, screen, pupilPoints, virtualImage)
f = time.clock()
print (f-s)/10.

iterations=13
factor = start 
factor = -0.037037037037037035 
u=4
v=4
cost - pos
eyePnt1 = pupilPoints[0]
eyePnt2 = pupilPoints[1]
virtualPnt = np.array(mc.xform('%s.vtx[0]'%vi,q=1,ws=1,t=1))
surf = mirror
spans = range(4,20,3)
bMirror = mirror
for i in spans:
    bMirror = mc.duplicate(bMirror,n='%sSpan%s'%(mirror,i))[0]
    mc.rebuildSurface(bMirror, ch=1, rpo=1,rt=0,end=1,kr=0,kcp=0,kc=0,su=i,sv=i,tol=9.223372037e+018,fr=0,dir=2)
    remap = mapImageInfluences(bMirror, screen, pupilPoints, virtualImage)
    bMirror = optimizeMirrorForImage(bMirror, screen, pupilPoints, virtualImage, remap, iterations=100*i)
    
spans = range(7,20,9)
bMirror = mirror
for i in spans:
    bMirror = mc.duplicate(bMirror,n='%sSpan%s'%(mirror,i))[0]
    mc.rebuildSurface(bMirror, ch=1, rpo=1,rt=0,end=1,kr=0,kcp=0,kc=0,su=i,sv=i,tol=9.223372037e+018,fr=0,dir=2)
    bMirror = optimizeCVsForImage(bMirror, screen, pupilPoints, virtualImage)        

bMirror = optimizeCVsForImage(bMirror, screen, pupilPoints, virtualImage, start=-0.004115226337448559, iterations=5, border=True)
bMirror = mc.duplicate(bMirror,n='%sSpan%s'%(mirror,7))[0]
mc.rebuildSurface(bMirror, ch=1, rpo=1,rt=0,end=1,kr=0,kcp=0,kc=0,su=7,sv=7,tol=9.223372037e+018,fr=0,dir=2)
bMirror = optimizeCVsForImage(bMirror, screen, pupilPoints, virtualImage, start=-0.037037037037037035, iterations=13)

np.sum(cost)
    
# setup some nodes
world = dg.SceneGraph()
screen = dg.Plane('screen', world, 4, [0,-1,0])
screen.setTranslate(*mc.xform('screen',q=1, ws=1, t=1))
screen.setRotate(*mc.xform('screen',q=1, ws=1, ro=1))
#screen.localPointToWorld(screen.point)
#screen.localVectorToWorld(screen.normal)
    
mirror = 'mirror1'
vi = 'virtualImage'
pupilPoints = [ # 4mm pupil
    np.array((-32.5,0.,80.)),
    np.array((-30.5,0.,80.)),
    np.array((-34.5,0.,80.)),
    np.array((-32.5,2,80.)),
    np.array((-32.5,-2,80.)),
    ]
points = reflectImagePlane(mirror, screen, pupilPoints, uRange=np.arange(.1,1,.1), vRange=np.arange(.1,1,.1), precision=3)
smoothness = calcSmoothness(points)

remap = mapInfluences(mirror, screen, pupilPoints, uRange=np.arange(.1,1,.1), vRange=np.arange(.1,1,.1))
points = optimizeMirrorShape(mirror, screen, pupilPoints, remap.astype(int), iterations=1)
drawPoly(points)
drawPoly(pPoints)
drawPoly(nPoints)
np.sum(smoothness)

points = optimizeMirrorShape(mirror, screen, pupilPoints, remap.astype(int), iterations=100, costFunction=calcDistance)
drawPoly(points)
mc.file(f=1, s=1)

rank = radiallySort(points.shape[:-1])
points = optimizeRankedMirrorShape(mirror, screen, pupilPoints, remap.astype(int), rank, iterations=100, costFunction=calcDistance)


reload(dg)
mc.xform(q=1,ws=1,t=1)

eyePnt1 = pupilPoints[0]
eyePnt2 = pupilPoints[1]
virtualPnt = np.array(mc.xform('%s.vtx[0]'%vi,q=1,ws=1,t=1))
surf = mirror
remap = mapImageInfluences(mirror, screen, pupilPoints, virtualImage)

iter = 10
s = time.clock()
for i in range(iter):
    pointSpreadCost(mirror, screen, pupilPoints, virtualImage)
f = time.clock()
print (f-s)/10.

    

targets = [
    np.array(mc.xform('mirror4Span7.cv[0][0]',q=1,ws=1,t=1)),
    np.array(mc.xform('mirror4Span7.cv[9][0]',q=1,ws=1,t=1)),
    np.array(mc.xform('mirror4Span7.cv[9][9]',q=1,ws=1,t=1)),
]
update = [
    np.array(mc.xform('mirror1Span7transformed.cv[0][0]',q=1,ws=1,t=1)),
    np.array(mc.xform('mirror1Span7transformed.cv[9][0]',q=1,ws=1,t=1)),
    np.array(mc.xform('mirror1Span7transformed.cv[9][9]',q=1,ws=1,t=1)),
]
ray1 = dg.Ray(targets[0],targets[1]-targets[0])
ray2 = dg.Ray(targets[1],targets[2]-targets[1])
goal = np.cross(ray1.vector, ray2.vector)
goal = goal/np.linalg.norm(goal)

ray1 = dg.Ray(update[0],update[1]-update[0])
ray2 = dg.Ray(update[1],update[2]-update[1])
move = np.cross(ray1.vector, ray2.vector)
move = move/np.linalg.norm(move)

v = np.cross(move, goal)
s = np.linalg.norm(v)
c = np.dot(move, goal)
vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])

r = np.identity(3)+vx+(vx**2*(1-c)/s**2)
mc.xform('mirror1Span7transformed',r=1, rotation=-xm.getRotation(np.matrix(r),[0,1,2]))







# setup some nodes
posi = mc.createNode('pointOnSurfaceInfo')
mc.connectAttr('nurbsPlaneShape2.worldSpace[0]', '%s.inputSurface'%posi)
grp = mc.group(em=1, n='rays')
reflect = mc.group(em=1, n='reflections', p=grp)
# get some info
eyeCenter = array(mc.xform('rtPupil',q=1,ws=1,t=1))
# calculate image plane
screen = g.Plane('screen', 4, [0,-1,0])
screen.setTranslate(*mc.xform('screen',q=1, ws=1, t=1))
screen.setRotate(*mc.xform('screen',q=1, ws=1, ro=1))
#screen.matrix
#screen.point
#screen.localPointToWorld(screen.point)
#screen.localVectorToWorld(screen.normal)

# Do a stochastic gradient descent
# define our degrees of freedom
obj = 'curvedLensGrp'
attr = ['tx']#,'ty','tz','rx','ry','rz']
origValues = []
for a in attr:
    origValues.append(mc.getAttr('%s.%s'%(obj,a)))
step = [.1]#,1,1,1,1,1]
variance = [] # for plotting
minValues = []

for idx, a in enumerate(attr):
    value = origValues[idx]-step[idx]*50
    minValue = None
    minVariance = float('inf')
    for s in range(101):
        mc.setAttr('%s.%s'%(obj,a), value)
        distances = []
        for u in [x/10.0 for x in range(11)]:
            for v in [x/10.0 for x in range(11)]:
                mc.setAttr('%s.parameterU'%posi, u)
                mc.setAttr('%s.parameterV'%posi, v)
                target = array(mc.getAttr('%s.position'%posi)[0])
                myRay = g.Ray(eyeCenter,target-eyeCenter)
                # reflect the ray
                normal = array(mc.getAttr('%s.normal'%posi)[0])
                reflectVect = myRay.vector-2*(np.dot(myRay.vector,normal))*normal
                reflectRay = g.Ray(target,reflectVect)
                # intersect ray with view plane
                intersection = screen.intersection(reflectRay)
                # draw the rays
                #crv = mc.curve(d=1, p=[eyeCenter,target])
                #mc.parent(crv, grp)
                if intersection != []:
                    # get distnace
                    distances.append(norm(target-eyeCenter)+intersection[0]['distance'])
                    #crv = mc.curve(d=1, p=[target,intersection[0]['point']])
                    #mc.parent(crv, reflect)
        var = max(distances) - min(distances)
        if var < minVariance:
            minVariance = var
            minValue = value
        variance.append(var)
        value += step[idx]
    mc.setAttr('%s.%s'%(obj,a), origValues[idx])
    
p = []
x=0
for y in variance:
    p.append((x,y,0))
    x+=1
mc.curve(p=p, n=attr[0])
mc.setAttr('%s.%s'%(obj,a), minValue)