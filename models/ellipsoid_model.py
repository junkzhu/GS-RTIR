from dataclasses import dataclass
import mitsuba as mi
import drjit as dr

@dataclass
class Ellipsoid:
    center:  mi.Point3f = mi.Point3f(0)
    scale:   mi.Vector3f = mi.Vector3f(0)
    quat:    mi.Quaternion4f = mi.Quaternion4f(0)
    rot:     mi.Matrix3f = mi.Matrix3f(0)
    extent:  mi.Float = mi.Float(3.0)
    
    @staticmethod
    def ravel(center, scale, quat):
        data = dr.empty(mi.Float, dr.width(center) * 10)
        idx = dr.arange(mi.UInt32, dr.width(center))
        for i in range(3):
            dr.scatter(data, center[i], idx * 10 + i)
        for i in range(3):
            dr.scatter(data, scale[i], idx * 10 + 3 + i)
        for i in range(4):
            dr.scatter(data, quat[i], idx * 10 + 6 + i)
        return data

    @staticmethod
    def unravel(data):
        idx = dr.arange(mi.UInt32, dr.width(data) // 10)
        center = mi.Point3f([dr.gather(mi.Float, data, idx * 10 + 0 + i) for i in range(3)])
        scale  = mi.Vector3f([dr.gather(mi.Float, data, idx * 10 + 3 + i) for i in range(3)])
        quat   = mi.Quaternion4f([dr.gather(mi.Float, data, idx * 10 + 6 + i) for i in range(4)])
        rot    = dr.quat_to_matrix(quat, 4)
        return Ellipsoid(center, scale, quat, rot, extent=None)