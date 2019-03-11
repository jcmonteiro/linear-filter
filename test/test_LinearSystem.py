import numpy as np
import yaml, unittest

import sys
sys.path.append('../build')
from linear_system import LinearSystem, IntegrationMethod

def initFilters(data):
    res = (LinearSystem(),LinearSystem(),LinearSystem())
    methods = [IntegrationMethod.TUSTIN, IntegrationMethod.FORWARD_EULER, IntegrationMethod.BACKWARD_EULER]
    init_input = data.input[0] * np.ones( (1, data.order) )
    for k in range(0,len(res)):
        res[k].setIntegrationMethod(methods[k])
        res[k].setSampling(data.sampling)
        res[k].setPrewarpFrequency(data.prewarp)
        res[k].setFilter(data.num, data.den)
        res[k].setInitialOutputDerivatives(data.ydy0)
        res[k].setInitialTime(0)
        res[k].discretizeSystem()
        res[k].setInitialStateMIMO(init_input)
    return res

class Data:
    def __init__(self, data_dict):
        self.samples = int(data_dict["n"])
        self.order = int(data_dict["order"])
        self.sampling = float(data_dict["Ts"])
        self.prewarp = float(data_dict["omega"])
        self.ydy0 = np.array(data_dict["ydy0"]).reshape( (1, self.order) )
        self.num = np.array(data_dict["num"])
        self.den = np.array(data_dict["den"])
        self.res_fwd = np.array(data_dict["y_fwd"])
        self.res_bwd = np.array(data_dict["y_bwd"])
        self.res_tustin = np.array(data_dict["y_tustin"])
        self.input = np.array(data_dict["u"])

class ProgressBar:
    def __init__(self, maxval = 100, width = 20):
        self.max = float(maxval)
        self.width = int(width)

    def update(self, value):
        if (value < 0):
            value = 0
        elif (value > self.max):
            value = self.max

        rate = value / self.max
        fmt = "[%-" + str(self.width) + "s] %d%%"

        # reset the cursor to the beginning of the line
        sys.stdout.write('\r')
        # print the progress bar
        sys.stdout.write(fmt % ('=' * int(rate * self.width), 100 * rate))
        # force everything to be printed
        sys.stdout.flush()

def computeError(vec1, vec2):
    return np.amax( np.absolute( vec1 - vec2 ) )

class TestLinearSystem(unittest.TestCase):
    def test_filtering(self):
        print("Loading YAML file...")
        stream = open("test_LinearSystem.yml", 'r')
        doc = yaml.load(stream)
        stream.close()
        print("Loaded!")

        tolerance = 1e-5
        progress = ProgressBar( len(doc) )
        n_info = 0
        for info in doc:
            n_info += 1
            progress.update(n_info)
            data = Data(info)
            filters = initFilters(data)

            res_filters = []
            for k in range (len(filters)):
                res_filters.append( np.zeros((1, data.samples)) )

            for k in range (0, data.samples):
                time = LinearSystem.getTimeFromSeconds( (k+1) * data.sampling )
                ind_filter = 0
                for filter in filters:
                    out = filter.update( np.array( data.input[k] ).reshape(1,1), time )
                    res_filters[ind_filter][0,k] = out
                    ind_filter += 1

            max_error = computeError(res_filters[0], data.res_tustin)
            self.assertTrue(max_error < tolerance)

            max_error = computeError(res_filters[1], data.res_fwd)
            self.assertTrue(max_error < tolerance)

            max_error = computeError(res_filters[2], data.res_bwd)
            self.assertTrue(max_error < tolerance)
        print("")

if __name__ == "__main__":
    unittest.main()