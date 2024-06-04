
def GetObsJacs(xPred, xFeature):
    jHxv = zeros((2,3))
    jHxf = zeros((2,2))
    Delta = (xFeature-xPred[0:2])
    r = norm(Delta)

    jHxv[0,0] = -Delta[0,0] / r
    jHxv[0,1] = -Delta[1,0] / r
    jHxv[1,0] = Delta[1,0] / (r**2)
    jHxv[1,1] = -Delta[0,0] / (r**2)
    jHxv[1,2] = -1
    jHxf[0:2,0:2] = -jHxv[0:2,0:2]
    return jHxv,jHxf

def GetNewFeatureJacs(Xv, z):
    r, a = z[0,0], z[1,0] + Xv[2,0]
    c, s = cos(a), sin(a)
    jGz = array([
        [c, -r*s],
        [s, r*c]
    ])
    jGx = array([
        [1, 0, -r*s],
        [0, 1, r*c]
    ])
    return jGx, jGz