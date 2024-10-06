import numpy as np

class Waves():

    ncomp = 0
    wave_amp = None
    wave_omg = None
    wave_dir = None
    wave_phs = None

    def __init__(self, wave_data=None):
        if wave_data is not None:

            self.wave_amp = []
            self.wave_omg = []
            self.wave_dir = []
            self.wave_phs = []

            for wave_type in wave_data:
                wave_comp = np.array(wave_type['data'])
                
                if wave_type['name'] == 'regular':
                    if wave_comp.ndim == 1:
                        self.ncomp += 1
                        self.wave_amp.append(wave_comp[0]/2)
                        self.wave_omg.append(2.0*np.pi/wave_comp[1])
                        self.wave_dir.append(wave_comp[2] * np.pi / 180)
                        self.wave_phs.append(wave_comp[3] * np.pi / 180)

                    elif wave_comp.ndim == 2:
                        self.ncomp += wave_comp.shape[0]
                        self.wave_amp.extend(wave_comp[:, 0] / 2)
                        self.wave_omg.extend(2.0 * np.pi / wave_comp[:, 1])
                        self.wave_dir.extend(wave_comp[:, 2] * np.pi / 180.0)
                        self.wave_phs.extend(wave_comp[:, 3] * np.pi / 180.0)
                
            self.wave_amp = np.array(self.wave_amp)
            self.wave_omg = np.array(self.wave_omg)
            self.wave_dir = np.array(self.wave_dir)
            self.wave_phs = np.array(self.wave_phs)