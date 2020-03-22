import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import sys

import noise

from scipy.spatial import Voronoi 

class VoronoiFunctions:

    def __init__(self, width, height, resolution= None, seed= 0):
        self.width = width
        self.height = height
        self.seed = seed
        self.resolution = resolution
        self.makeRandomPoints()
        self.buildVoronoiCells()

    # should only be run before plotting, use standard Voronoi() to generate regions
    @staticmethod
    def voronoiFinitePloygons2D(vor, radius= None):
        # Pauli Virtanen, author #
        # https://gist.github.com/pv/8036995 # 
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.
        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.
        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.
        """

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()*2

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2] - vor.points[p1] # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices) 

    def makeRandomPoints(self):
        # generate seeds
        # x and y handled seperately to easily scale with window size
        np.random.seed(self.seed)
        seedX = (np.random.rand(self.resolution) * 2 - 1) * 0.5 * self.width
        seedY = (np.random.rand(self.resolution) * 2 - 1) * 0.5 * self.height

        self.points = np.stack((seedX, seedY), axis = -1)

    def buildVoronoiCells(self):
        eps = sys.float_info.epsilon
        
        self.voronoi = Voronoi(self.points)

        self.filteredRegions = [] # list of regions with vertices inside Voronoi map
        for region in self.voronoi.regions:
            inside_map = True    # is this region inside the Voronoi map?
            for index in region: # index = the idx of a vertex in the current region

                # check if index is inside Voronoi map (indices == -1 are outside map)
                if index == -1:
                    inside_map = False
                    break

                # check if the current coordinate is in the Voronoi map's bounding box
                # else:
                #     coords = self.voronoi.vertices[index]
                #     if not (-0.5 * self.width - eps <= coords[0] and
                #             0.5 * self.width + eps >= coords[0] and
                #             -0.5 * self.height - eps <= coords[1] and
                #             0.5 * self.height + eps >= coords[1]):
                #         inside_map = False
                #         break

            # store the region if it has vertices and is inside Voronoi map
            if region != [] and inside_map:
                self.filteredRegions.append(region)

    def findCentroid(self, vertices):
        # Douglas Duhaime, author #
        # https://gist.github.com/duhaime/3e781194ebaccc28351a5d53989caa70 #
        '''
        Find the centroid of a Voroni region described by `vertices`, and return a
        np array with the x and y coords of that centroid.
        The equation for the method used here to find the centroid of a 2D polygon
        is given here: https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
        @params: np.array `vertices` a numpy array with shape n,2
        @returns np.array a numpy array that defines the x, y coords
        of the centroid described by `vertices`
        '''
        area = 0
        centroid_x = 0
        centroid_y = 0
        for i in range(len(vertices)-1):
            step = (vertices[i, 0] * vertices[i+1, 1]) - (vertices[i+1, 0] * vertices[i, 1])
            area += step
            centroid_x += (vertices[i, 0] + vertices[i+1, 0]) * step
            centroid_y += (vertices[i, 1] + vertices[i+1, 1]) * step
        area /= 2
        centroid_x = (1.0/(6.0*area)) * centroid_x
        centroid_y = (1.0/(6.0*area)) * centroid_y
        return np.array([[centroid_x, centroid_y]])

    def relaxPoints(self):
        # Douglas Duhaime, author #
        # https://gist.github.com/duhaime/3e781194ebaccc28351a5d53989caa70 #
        '''
        Moves each point to the centroid of its cell in the Voronoi map to "relax"
        the points (i.e. jitter them so as to spread them out within the space).
        '''
        centroids = []
        for region in self.filteredRegions:
            vertices = self.voronoi.vertices[region + [region[0]], :]
            centroid = self.findCentroid(vertices) # get the centroid of these verts
            centroids.append(list(centroid[0]))

        self.points = centroids # store the centroids as the new point positions
        self.buildVoronoiCells() # build new cells from new points

    def plotVoronoi(self):
            plt.figure()
            ax = plt.subplot(111)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_xlim(-0.5 * self.width, 0.5 * self.width)
            ax.set_ylim(-0.5 * self.height, 0.5 * self.height)

            regions, vertices = VoronoiFunctions.voronoiFinitePloygons2D(self.voronoi)
            voronoiPolygons = [vertices[reg] for reg in regions]

            for polygon in voronoiPolygons:
                colored_cell = Polygon(polygon)
                ax.add_patch(colored_cell)

            plt.show()

class TerrainModification:
    def __init__(self, width, height, resolution, seed):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.seed = seed
        
    def generateNoiseMap(self, scale, octaves= 4, persistence= 0.5, lacunarity= 2.0):
        noiseMap = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                noiseMap[x][y] = noise.snoise2(x / scale, y / scale, octaves, persistence, lacunarity, base= self.seed)
        return noiseMap
    
    
    def plotHeightmap(self, heightMap):
    # Plot #
        plt.imshow(heightMap)
        plt.show()

    def erodeTerrain(self, heightMap, interations):
        pass

width = 1000
height = 1000
points = 1000
seed = 89

scale = 800
octaves = 4
lacunarity = 2
persistence = .5

voronoiTerrain = VoronoiFunctions(width, height, points, seed)
voronoiTerrain.plotVoronoi()
voronoiTerrain.relaxPoints()
voronoiTerrain.plotVoronoi()
voronoiTerrain.relaxPoints()
voronoiTerrain.plotVoronoi()
voronoiTerrain.relaxPoints()
voronoiTerrain.plotVoronoi()

#noiseMap = terrain.generateNoiseMap(scale, octaves, persistence, lacunarity, seed)
#terrain.plotHeightmap(noiseMap)

