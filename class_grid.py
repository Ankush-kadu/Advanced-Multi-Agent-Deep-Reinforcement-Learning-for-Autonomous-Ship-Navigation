import numpy as np

class Grid:

    def __init__(self, gdf_fname = None, start_location=None, start_orientation=None, scale=1):
        self.GDF_file = gdf_fname
        self.scale = scale
        self.ulen = 0.0
        self.gravity = 0.0
        self.npan = 0        
        self.Vertices = None
        self.Centroids = None
        self.Diagonals = None
        self.UnitVecs = None
        self.Area = None

        self.start_location = start_location
        self.start_orientation = start_orientation
        
        self.x_sec = None
        self.B_sec = None
        self.T_sec = None        
        self.Displacement = None
        self.WettedArea = None
               
    def load_gdf(self):

        resultList = []
        f = open(self.GDF_file, 'r')
        count = 0

        for line in f:
            count += 1
            if (count > 4 and count < 4*self.npan+5):
                 fVals = np.array([float(e) for e in line.split()])
                 resultList.append(fVals)
            elif(count==2):
                self.ulen = float(line.split()[0])
                self.gravity = float(line.split()[1])
            elif(count==4):
                self.npan = int(line)
        f.close()

        # By default it is assumed that the GDF file has vertex order that results in 
        # normals of the mesh pointing out of the vehicle.

        # Read in vertices with opposite order (should result in inward normals)
        self.Vertices = np.reshape(np.array(resultList),(-1,4,3))[:, [0, 3, 2, 1], :] / self.scale
        
        # Calculate centroids
        d = np.sqrt(np.sum(np.square(np.roll(self.Vertices,[0,-1,0],axis=(0,1,2))-self.Vertices),axis=2))
        dinv = 1/np.sum(d,axis=1)        
        cen = (np.roll(self.Vertices,[0,-1,0],axis=(0,1,2))+self.Vertices)/2.0
        self.Centroids = np.einsum('ij,i->ij',np.einsum('i...k,ikl->il', d, cen),dinv)

        # Calculate Diagonals
        self.Diagonals = np.concatenate((self.Vertices[:,2,:]-self.Vertices[:,0,:],self.Vertices[:,3,:]-self.Vertices[:,1,:]),axis=1).reshape(self.npan,2,3)

        # Calculate unitvectors
        N = np.cross(self.Vertices[:,2,:]-self.Vertices[:,0,:],self.Vertices[:,3,:]-self.Vertices[:,1,:])
        mid = cen[:,2,:]-cen[:,0,:] #<---l
        n = np.multiply(N,np.array([1/np.linalg.norm(N,axis=1),]*3).T)
        l = np.multiply(mid,np.array([1/np.linalg.norm(mid,axis=1),]*3).T)
        m = np.cross(n,l)
        self.UnitVecs = np.concatenate((l,m,n),axis=1).reshape(self.npan,3,3)

        # Calculate panel areas
        self.Area = np.linalg.norm(N,axis=1)/2.0

        # Calcualte displacement
        h = (self.Centroids[:, 2] + self.start_location[2])
        self.Displacement = np.sum(-self.Area * (self.UnitVecs[:, 2, 2] * h), where= h <= 0)
        self.WettedArea = np.sum(self.Area, where= h <= 0)


    def create_2d_sections(self):
        if self.start_location is None:
            self.start_location = np.zeros(3)
        
        if self.start_orientation is None:
            self.start_orientation = np.zeros(4)
        
        z_th = -self.start_orientation[2]
        
        lwl = np.max(self.Vertices[:, :, 0]) - np.min(self.Vertices[:, :, 0])
        x = np.linspace(-0.475*lwl, 0.475*lwl, 100)

        tol = 1e-3
        count = 0
        
        x_ray_y = []
        y_ray_y = []
        
        x_ray_z = []
        z_ray_z = []

        for x_ray in x.tolist():
            origin = np.array([x_ray, tol, -self.start_location[2]-tol])
            direction_y = np.array([0, 1, 0])
            direction_z = np.array([0, 0, 1])
            int_ind_y, int_pts_y = ray_intersects_quadrilateral(origin, direction_y, self.Vertices)
            int_ind_z, int_pts_z = ray_intersects_quadrilateral(origin, direction_z, self.Vertices)
            count += 1
            
            if any(int_ind_y):
                x_ray_y.append(x_ray)
                y_ray_y.append(int_pts_y[int_ind_y][0, 1])
                # print(count, x_ray, int_pts_y[int_ind_y][0, 1])                
            
            if any(int_ind_z):
                x_ray_z.append(x_ray)
                z_ray_z.append(int_pts_z[int_ind_z][0, 2])
                # print(count, x_ray, int_pts_z[int_ind_z][0, 2])

        x_ray_y = np.array(x_ray_y)
        y_ray_y = np.array(y_ray_y)

        x_ray_z = np.array(x_ray_z)
        z_ray_z = np.array(z_ray_z)

        x_ray = np.intersect1d(x_ray_y, x_ray_z)
        y_ray = y_ray_y[np.isin(x_ray_y, x_ray)]
        z_ray = z_ray_z[np.isin(x_ray_z, x_ray)]

        B_ray = 2*np.abs(y_ray)
        T_ray = np.abs(z_ray)

        hoener = B_ray/2/T_ray

        self.x_sec = x_ray
        self.B_sec = B_ray
        self.T_sec = T_ray        
        
def ray_intersects_quadrilateral(origin, direction, quad_points):
    # Extract quadrilateral points
    p0 = quad_points[:, 0, :]
    p1 = quad_points[:, 1, :]
    p2 = quad_points[:, 2, :]
    p3 = quad_points[:, 3, :]

    # Define vectors for the edges of the quadrilateral
    e1 = p1 - p0
    e2 = p2 - p1
    e3 = p3 - p2
    e4 = p0 - p3

    d1 = p2 - p0
    d2 = p3 - p1

    # Calculate the normal vector of the quadrilateral
    normal = 0.5 * np.cross(d1, d2)

    # Calculate the determinant of the direction and normal
    det = np.dot(normal, direction)
    
    # Initialize intersection points
    tol = 1e-6
    intersection_boolean = det > tol

    # Calculate the distance from the origin to the plane of the quadrilateral
    d = np.where(intersection_boolean, np.sum((p0 - origin) * normal, axis=1) / det, 1e-8)
    
    # Calculate the point of intersection
    intersection_point = origin + d[:, np.newaxis] @ direction[np.newaxis, :]

    # Check if the intersection point is inside the quadrilateral
    ind = np.logical_and(np.logical_and(np.sum(np.cross(e1, intersection_point - p0) * normal, axis=1) >= 0, \
            np.sum(np.cross(e2, intersection_point - p1) * normal, axis=1) >= 0), \
            np.logical_and(np.sum(np.cross(e3, intersection_point - p2) * normal, axis=1) >= 0, \
            np.sum(np.cross(e4, intersection_point - p3) * normal, axis=1) >= 0))
    
    ind = np.where(intersection_boolean, ind, False)

    intersection_boolean = ind
    intersection_point[:, 0] = np.where(ind, intersection_point[:, 0], 0)
    intersection_point[:, 1] = np.where(ind, intersection_point[:, 1], 0)
    intersection_point[:, 2] = np.where(ind, intersection_point[:, 2], 0)
    
    return intersection_boolean, intersection_point



        
        

