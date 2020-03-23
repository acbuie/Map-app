import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

import noise

from scipy.spatial import Voronoi

# matplotlib.use('GTK3Agg')


class VoronoiFunctions:

    def __init__(self, width, height, resolution=None, seed=0):
        self.width = width
        self.height = height
        self.seed = seed
        self.resolution = resolution
        self.makeRandomPoints()
        self.buildVoronoiCells()

    def makeRandomPoints(self):
        # generate seeds
        # x and y handled seperately to easily scale with window size
        np.random.seed(self.seed)
        seedX = (np.random.rand(self.resolution) * 2 - 1) * 0.5 * self.width
        seedY = (np.random.rand(self.resolution) * 2 - 1) * 0.5 * self.height

        self.points = np.stack((seedX, seedY), axis=-1)

        print('Done makeRandomPoints')

    def buildVoronoiCells(self):
        """
        Generates the voronoi object. For some reason Voronoi() generates an empty region, sometimes at the front
        of self.voronoi.regions, sometimes at the end. So, also removes this region from self.voronoi.regions.
        """
        # create voronoi object
        self.voronoi = Voronoi(self.points)

        # remove empty region from self.voronoi.regions
        # see about doing with with np arrays?
        for index, ls in enumerate(self.voronoi.regions):
            if Utility.isEmptyList(ls):
                del self.voronoi.regions[index]
                break

        # convert self.voronoi.regions to array of array instead of list of list for faster calculation
        self.voronoi.regions = np.array(
            [np.array(region) for region in self.voronoi.regions]
        )

        # create voronoi regions object
        self.vorPolygon = [self.voronoi.vertices[reg]
                           for reg in self.voronoi.regions]

        print('Done buildVoronoiCells')

    def improveRandomPoints(self, iterations):
        """
        Takes self and averages vertices of self.voronoi.vertices for each voronoi region.
        The average of these vertices is stored in self.points. This is an opproximation of Lloyd relaxation.
        """
        for iteration in range(iterations):
            avgPoints = []
            # potential to vectorize?
            # currently loops over vertices[regions] twice (once in buildVoronoiCells)
            # but I struggled to average the points from the vorPolygon object. Will try again another day.
            for reg in self.voronoi.regions:
                vorPoly = self.voronoi.vertices[reg]
                avgPoints.append(list(np.mean(vorPoly, 0)))

            # set self.points to new point, regenerate Voronoi cells
            self.points = avgPoints

            print(f'Done improveRandomPoints {iteration}')
            self.buildVoronoiCells()


class ElevationFunctions:

    @staticmethod
    def simplexNoise(point, seed=0, scale=100, octaves=4, persistence=0.5, lacunarity=2.0):
        """
        Return a noise value given an (x, y), and other optional inputs
        """
        return noise.snoise2(
            point[0] / scale, point[1] / scale, octaves, persistence, lacunarity, base=seed
        )

    @staticmethod
    def generateRasterPoints(width, height):
        """
        Method to create evenly spaced points in array. Takes int x, int y

        For generataing a noiseMap, instead of a voronoi diagram, if wanted
        Not sure if this will be used much. Currently needs to be reshaped
        after generating elevationFromNoise. Could probably refactor the code
        to run a different noise method on the correcly shaped data?
        """
        return np.indices((width + 1, height + 1)).T.reshape(-1, 2)

    @staticmethod
    def elevationFromNoise(points):
        """
        Apply noise function to voronoi points. Takes array of (x, y), or meshgrid arrays
        Type 'voronoi' applies function to 1-D array of (x, y), type 'raster' applies function
        to 2-D arrays
        Replace 'ElevationFunctions.simplexNoise' with other elevation functions
        """
        noise = np.apply_along_axis(ElevationFunctions.simplexNoise, 1, points)
        normalized = (noise - np.amin(noise)) / \
            (np.amax(noise) - np.amin(noise))

        print('Done elevationFromNoise')

        return normalized

    def erodeTerrain(self, heightMap, interations):
        pass


class PlottingFunctions:

    @staticmethod
    def colorFromElevation(elevation):
        """
        Takes a value (elevation) and returns a color
        Currently just sets a random color
        """
        # Possible colors to be, would like to set up a selector eventually

        darkblue = [10, 100, 180]
        lightblue = [65, 105, 225]
        beach = [238, 214, 175]
        green = [34, 139, 34]
        mountain = [55, 60, 65]
        snow = [255, 250, 250]

        if elevation < 0.30:
            color = darkblue
        elif elevation < 0.47:
            color = lightblue
        elif elevation < 0.55:
            color = beach
        elif elevation < 0.80:
            color = green
        elif elevation < 0.95:
            color = mountain
        elif elevation <= 1.00:
            color = snow

        return list(np.array(color)/255)

    @staticmethod
    def plotVoronoi(width, height, vorPolygon, elevation):
        """
        Takes object of class 'VoronoiFunctions' and elevation array
        """
        plt.figure()
        ax = plt.subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(-0.5 * width, 0.5 * width)
        ax.set_ylim(-0.5 * height, 0.5 * height)

        for index, polygon in enumerate(vorPolygon):
            colored_cell = Polygon(
                polygon,
                facecolor=PlottingFunctions.colorFromElevation(
                    elevation[index]),
                linewidth=None
            )
            ax.add_patch(colored_cell)

        plt.show()


class Utility:

    @staticmethod
    def isEmptyList(list):
        """
        Returns true if a list is empty
        """
        if not list:
            return True
        else:
            return False


width = 1000
height = 1000
points = 10000
seed = 89


# construct voronoiTerrain
voronoiTerrain = VoronoiFunctions(width, height, points, seed)
# Lloyd relaxation of points
voronoiTerrain.improveRandomPoints(2)

# Calculate elevation for each voronoi region
elevationVor = ElevationFunctions.elevationFromNoise(voronoiTerrain.points)

PlottingFunctions.plotVoronoi(
    width, height, voronoiTerrain.vorPolygon, elevationVor
)
