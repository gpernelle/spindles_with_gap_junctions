from fns.utils import *
import tensorflow as tf

'''
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

'''

def diag(W):
    '''
    Set diagonal elements to 0
    '''
    return W - np.diag(np.diag(W))


class TRN:
    def __init__(self,
                 T=100,
                 config=None,
                 profiling=False
                 ):
        # reset the tensorflow graph
        tf.reset_default_graph()

        ## simulation parameters
        # for profiling
        self.profiling = profiling
        # to save raster plot
        self.spikeMonitor = False
        self.debug = 0
        # sampling interval of the weight matrices
        self.weight_step = 10
        # sampling interval for monitoring variables
        self.monitor_step = 1
        # to save individual traces
        self.monitor_single = False

        # default neurons distribution

        self.NE1 = 200
        self.NE2 = 0
        self.NI1 = 100
        self.NI2 = 0

        self.N = self.NE1 + self.NI1 + self.NE2 + self.NI2

        # number of timesteps and timestep in ms (simulation duration in ms = T*dt)
        self.T = T
        self.dt = 1 ## for smaller dt, increase sligthly q_thresh and/or decrease nu for similar results

        # time constant for 1st subnet
        self.tauv1 = 17
        # time constant for 2nd subnet
        self.tauv2 = 17
        # time constant for the adaption
        self.tau_u = 10
        # adaptation coupling parameters
        self.u_a = 10

        # IZH neuron parameters
        self.mod_a = 60
        self.mod_b = 15
        self.mod_c = 4
        # neuron reset values
        self.v_r_I = -60
        self.v_r_E = -70
        # neuron threshold values
        self.v_thresh_E = 0
        self.v_thresh_I = 25
        # IAF neuron time constant
        self.tau_v_E = 40
        # IAF neuron resistance
        self.Rm = 0.6
        # inhibitory neuron time constant
        self.tau_v_I = 8
        # v initialisation
        self.v_init_mean = -100
        self.v_init_std = 30

        # synapses parameters
        self.tau_I_I = 10
        self.tau_I_E = 12

        ## bursting filter
        self.tau_burst = 8.0
        self.burst_thresh = 1.3

        ## model with spindles
        self.spindles = True
        self.spindles_rule = "bursting"
        self.tau_q = 6000
        self.q_thresh = 0.3
        self.alpha = self.dt / (self.tau_q + self.dt)

        ## plasticity variables
        # plasticity multiplier
        self.FACT = 1 / self.dt
        # LTD learning rate
        self.alpha_LTD = 1.569e-5
        # LTP/LTD ratio
        self.ratio = 15
        # time during which the plasticity is turned of after subnetworks are connected
        self.stabTime = int(1000/self.dt)
        # time when to stop the plasticity
        self.stopTime = np.inf
        # time at which to connect subnetworks
        self.connectTime = 0

        ## input parameters
        # mean input current to inhibitory neurons
        self.nu = 40
        self.sigmaNoise = 100

        # extra current to excitatory neurons
        self.kInputE1 = 0
        self.kInputE2 = 0
        self.kNoiseE1 = 1
        self.kNoiseE2 = 1
        self.noiseScaling = 1 / (1 / (2 * 2 / self.dt)) ** 0.5 * self.sigmaNoise

        # default input signal
        self.input = 0

        ## connectivity
        # slope of the WII curve
        self.k = 40

        self.wE1E1 = 500
        self.wE1E2 = 0
        self.wE1I1 = 300
        self.wE1I2 = 0

        self.wE2E1 = 0
        self.wE2E2 = 0
        self.wE2I1 = 0
        self.wE2I2 = 0

        self.wI1E1 = -1000
        self.wI1E2 = 0
        self.wI1I1 = -200
        self.wI1I2 = 0

        self.wI2E1 = 0
        self.wI2E2 = 0
        self.wI2I1 = 0
        self.wI2I2 = 0

        # LTP softbound
        self.g0 = 13

        # gap junction conductances
        self.g1 = 0.5
        self.g2 = 0
        # proportion of gap junction to delete
        self.propToDelete = 0

        self.v0 = 1

        # number of shared gap junctions between subnets 1 and 2
        self.sG = 0

        # random distribution parameters
        self.distrib = 'lognormal_gap'
        self.mu = 1
        self.sigma = 1

        ## tensorflow session parameters
        gpu_options = tf.GPUOptions(  # per_process_gpu_memory_fraction=memfraction,
            allow_growth=True)

        config = tf.ConfigProto(
            log_device_placement=False,
            inter_op_parallelism_threads=0,
            intra_op_parallelism_threads=0,
            gpu_options=gpu_options
        )
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        self.sess = tf.InteractiveSession(config=config)

        if profiling:
            self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()
        else:
            self.run_metadata = None
            self.run_options = None

        self.double_thresh = False

    def updateVars(self):
        self.noiseScaling = tf.constant(1 / (1 / (2 * 2 / self.dt)) ** 0.5 * self.sigmaNoise, dtype=tf.float32)
        self.alpha = tf.constant(self.dt / (self.tau_q + self.dt), dtype=tf.float32)

    def makeVect(self):

        NE1, NE2, NI1, NI2 = self.NE1, self.NE2, self.NI1, self.NI2

        vConnE1 = np.concatenate([np.ones((NE1, 1)), np.zeros((NI1 + NE2 + NI2, 1))])
        vConnI1 = np.concatenate([np.zeros((NE1, 1)), np.ones((NI1, 1)), np.zeros((NE2 + NI2, 1))])
        vConnI2 = np.concatenate([np.zeros((NE1 + NI1, 1)), np.ones((NI2, 1)), np.zeros((NE2, 1))])
        vConnE2 = np.concatenate([np.zeros((NE1 + NI1 + NI2, 1)), np.ones((NE2, 1))])

        VE1 = tf.Variable(vConnE1, dtype='float32')
        VE2 = tf.Variable(vConnE2, dtype='float32')
        VI1 = tf.Variable(vConnI1, dtype='float32')
        VI2 = tf.Variable(vConnI2, dtype='float32')

        return VE1, VE2, VI1, VI2

    def add_shared_gap(self, W_, n):
        W = W_.copy()
        NE1, NE2, NI1, NI2 = self.NE1, self.NE2, self.NI1, self.NI2
        N1 = NE1 + NI1

        W[NE1:N1, N1:N1 + n] = 1
        W[N1:N1 + NI2, N1 - n:N1] = 1
        W[N1:N1 + n, NE1:N1] = 1
        W[N1 - n:N1, N1:N1 + NI2] = 1
        return diag(W)

    def makeConn(self, TF=True, distrib='lognormal_gap', mu=1, sigma=1, sG=0,
                 we1e1=1, we1e2=1, we1i1=1, we1i2=1,
                 we2e1=1, we2e2=1, we2i1=1, we2i2=1,
                 wi1e1=1, wi1e2=1, wi1i1=1, wi1i2=1,
                 wi2e1=1, wi2e2=1, wi2i1=1, wi2i2=1,
                 g1=1, g2=1, gS=1):

        NE1, NE2, NI1, NI2 = self.NE1, self.NE2, self.NI1, self.NI2
        N1 = NE1 + NI1
        N2 = NE2 + NI2
        # total number of neurons
        N = N1 + N2

        W0 = np.zeros((N1 + N2, N1 + N2))

        ## From E1
        # WE1E1
        WE1E1 = W0.copy()
        WE1E1[:NE1, :NE1] = we1e1
        WE1E1 = diag(WE1E1)

        # WE1E2
        WE1E2 = W0.copy()
        WE1E2[:NE1, -NE2:] = we1e2
        WE1E2 = WE1E2.T

        # WE1I1
        WE1I1 = W0.copy()
        WE1I1[:NE1, NE1:N1] = we1i1
        WE1I1 = diag(WE1I1)
        WE1I1 = WE1I1.T

        # WE1I2
        WE1I2 = W0.copy()
        WE1I2[:NE1, N1:N1 + NI2] = we1i2
        WE1I2 = diag(WE1I2)
        WE1I2 = WE1I2.T

        ## From E2
        # WE2E1
        WE2E1 = W0.copy()
        WE2E1[-NE2:, :NE1] = we2e1
        WE2E1 = WE2E1.T

        # WE2E2
        WE2E2 = W0.copy()
        WE2E2[-NE2:, -NE2:] = we2e2
        WE2E2 = diag(WE2E2)

        # WE2I1
        WE2I1 = W0.copy()
        WE2I1[-NE2:, NE1:N1] = we2i1
        WE2I1 = WE2I1.T

        # WE2I2
        WE2I2 = W0.copy()
        WE2I2[-NE2:, N1:N1 + NI2] = we2i2
        WE2I2 = diag(WE2I2)
        WE2I2 = WE2I2.T

        ## From I1
        # WI1E1
        WI1E1 = W0.copy()
        WI1E1[NE1:N1, :NE1] = wi1e1
        WI1E1 = diag(WI1E1)
        WI1E1 = WI1E1.T

        # WI1E2
        WI1E2 = W0.copy()
        WI1E2[NE1:N1, -NE2:] = wi1e2
        WI1E2 = diag(WI1E2)
        WI1E2 = WI1E2.T

        # WI1I1
        WI1I1 = W0.copy()
        WI1I1[NE1:N1, NE1:N1] = wi1i1
        WI1I1 = diag(WI1I1)

        # WI1I2
        WI1I2 = W0.copy()
        WI1I2[NE1:N1, N1:N1 + NI2] = wi1i2
        WI1I2 = diag(WI1I2)
        WI1I2 = WI1I2.T

        ## From I2
        # WI2E1
        WI2E1 = W0.copy()
        WI2E1[N1:N1 + NI2, 0:NE1] = wi2e1
        WI2E1 = diag(WI2E1)
        WI2E1 = WI2E1.T

        # WI2E2
        WI2E2 = W0.copy()
        WI2E2[N1:N1 + NI2, -NE2:] = wi2e2
        WI2E2 = diag(WI2E2)
        WI2E2 = WI2E2.T

        # WI2I1
        WI2I1 = W0.copy()
        WI2I1[N1:N1 + NI2, NE1:N1] = wi2i1
        WI2I1 = diag(WI2I1)
        WI2I1 = WI2I1.T

        # WI2I2
        WI2I2 = W0.copy()
        WI2I2[N1:N1 + NI2, N1:N1 + NI2] = wi2i2
        WI2I2 = diag(WI2I2)

        ## Gap junctions
        # WIIg1 gap junctions subnet1
        WIIg1 = W0.copy()
        WIIg1[NE1:N1, NE1:N1] = g1
        WIIg1 = diag(WIIg1)

        # WIIg2 gap junctions subnet1
        WIIg2 = W0.copy()
        WIIg2[N1:N1 + NI2, N1:N1 + NI2] = g2
        WIIg2 = diag(WIIg2)

        # shared Gap Junctions WIIg:
        WIIgS = self.add_shared_gap(W0, sG) * gS

        listmat = [WE1E1, WE1E2, WE1I1, WE1I2]
        listmat += [WE2E1, WE2E2, WE2I1, WE2I2]
        listmat += [WI1E1, WI1E2, WI1I1, WI1I2]
        listmat += [WI2E1, WI2E2, WI2I1, WI2I2]
        listmatG = [WIIg1, WIIg2, WIIgS]
        listmatAll = listmat + listmatG
        connMat = []
        if distrib == 'lognormal':
            for mat in listmatAll:
                mat = mat * np.random.lognormal(mu, sigma, (N1 + N2, N1 + N2))
                connMat.append(mat)

        elif distrib == 'uniform':
            for mat in listmatAll:
                mat = mat * np.random.random((N1 + N2, N1 + N2))
                connMat.append(mat)

        elif distrib == 'lognormal_gap':
            for mat in listmat:
                connMat.append(mat)

            for mat in listmatG:
                mat = mat * np.random.lognormal(mu, sigma, (N1 + N2, N1 + N2))
                connMat.append(mat)

        else:
            connMat = listmatAll

        WE1E1, WE1E2, WE1I1, WE1I2, \
        WE2E1, WE2E2, WE2I1, WE2I2, \
        WI1E1, WI1E2, WI1I1, WI1I2, \
        WI2E1, WI2E2, WI2I1, WI2I2, \
        WIIg1, WIIg2, WIIgS = connMat

        # get matrix of deleted connections (0s)
        A = np.random.rand(N, N)
        A = np.tril(A) + np.tril(A, -1).T
        connDelete = (A > self.propToDelete) * 1

        if TF:
            WE1E1 = tf.Variable(WE1E1, dtype=tf.float32, name='E1E1')
            WE1E2 = tf.Variable(WE1E2, dtype=tf.float32, name='E1E2')
            WE1I1 = tf.Variable(WE1I1, dtype=tf.float32, name='E1I1')
            WE1I2 = tf.Variable(WE1I2, dtype=tf.float32, name='E1I2')

            WE2E1 = tf.Variable(WE2E1, dtype=tf.float32, name='E2E1')
            WE2E2 = tf.Variable(WE2E2, dtype=tf.float32, name='E2E2')
            WE2I1 = tf.Variable(WE2I1, dtype=tf.float32, name='E2I1')
            WE2I2 = tf.Variable(WE2I2, dtype=tf.float32, name='E2I2')

            WI1E1 = tf.Variable(WI1E1, dtype=tf.float32, name='I1E1')
            WI1E2 = tf.Variable(WI1E2, dtype=tf.float32, name='I1E2')
            WI1I1 = tf.Variable(WI1I1, dtype=tf.float32, name='I1I1')
            WI1I2 = tf.Variable(WI1I2, dtype=tf.float32, name='I1I2')

            WI2E1 = tf.Variable(WI2E1, dtype=tf.float32, name='I2E1')
            WI2E2 = tf.Variable(WI2E2, dtype=tf.float32, name='I2E2')
            WI2I1 = tf.Variable(WI2I1, dtype=tf.float32, name='I2I1')
            WI2I2 = tf.Variable(WI2I2, dtype=tf.float32, name='I2I2')

            WIIg1 = tf.Variable(WIIg1, dtype=tf.float32, name='IIg1')
            WIIg2 = tf.Variable(WIIg2, dtype=tf.float32, name='IIg2')
            WIIgS = tf.Variable(WIIgS, dtype=tf.float32, name='IIgS')

            connDelete = tf.Variable(connDelete, dtype=tf.float32, name='connDelete')

            tf.global_variables_initializer().run()

        return WE1E1, WE1E2, WE1I1, WE1I2, \
               WE2E1, WE2E2, WE2I1, WE2I2, \
               WI1E1, WI1E2, WI1I1, WI1I2, \
               WI2E1, WI2E2, WI2I1, WI2I2, \
               WIIg1, WIIg2, WIIgS, connDelete

    def init_float(self, shape, name):
        return tf.Variable(tf.zeros(shape), name=name)

    def runTFSimul(self):
        #################################################################################
        ### INITIALISATION
        #################################################################################
        T = self.T
        dt = self.dt
        NE1, NE2, NI1, NI2 = self.NE1, self.NE2, self.NI1, self.NI2
        N1 = NE1 + NI1
        N2 = NE2 + NI2
        N = N1 + N2

        with tf.name_scope('spiking_bursting'):
            LowSp = self.init_float([N, 1], 'bursting')
            q = self.init_float([N, 1], 'spindles')
            qq = self.init_float([N, 1], 'spindles')
            p = self.init_float([N, 1], 'dep_on')
            q_ = self.init_float([N, 1], 'spindles')
            qq_ = self.init_float([N, 1], 'spindles')
            vv = self.init_float([N, 1], 'spiking')

        with tf.name_scope('monitoring'):
            # variables for monitoring
            ### sampling
            weight_step = self.weight_step
            monitor_step = self.monitor_step

            vvmE1 = self.init_float([T // monitor_step], "vvE1")
            vvmE2 = self.init_float([T // monitor_step], "vvE2")
            vvmI1 = self.init_float([T // monitor_step], "vvI1")
            vvmI2 = self.init_float([T // monitor_step], "vvI2")

            vmE1 = self.init_float([T // monitor_step], "vE1")
            vmE2 = self.init_float([T // monitor_step], "vE2")
            vmI1 = self.init_float([T // monitor_step], "vI1")
            vmI2 = self.init_float([T // monitor_step], "vI2")

            imE1 = self.init_float([T // monitor_step], "i1E1")
            imE2 = self.init_float([T // monitor_step], "i2E2")
            imI1 = self.init_float([T // monitor_step], "i1I1")
            imI2 = self.init_float([T // monitor_step], "imI2")

            pmI1 = self.init_float([T // monitor_step], "pm")
            qmI1 = self.init_float([T // monitor_step], "qm")
            qqmI1 = self.init_float([T // monitor_step], "qm")
            lowspmI1 = self.init_float([T // monitor_step], "lowspm")

            iGapm = self.init_float([T // monitor_step], "iGap")

            ### debugging
            Am = self.init_float([T // monitor_step], "Am")
            Bm = self.init_float([T // monitor_step], "Bm")
            dwm = self.init_float([T // monitor_step], "dwm")

            WI1I1m = self.init_float([T // weight_step], "WI1I1m")
            g1m = self.init_float([T // weight_step], "gamma_N1")
            g2m = self.init_float([T // weight_step], "gamma_N2")
            gSm = self.init_float([T // weight_step], "gamma_NS")

            if self.spikeMonitor:
                spikes = self.init_float([T, N], "spikes")
            else:
                spikes = self.init_float([1, N], "spikes")
            if self.monitor_single:
                iAll = self.init_float([T, N], "iAll")
                iChemAll = self.init_float([T, N], "iChem")
                vAll = self.init_float([T, N], "vAll")
                kAll = self.init_float([T, N], "kAll")
                qAll = self.init_float([T, N], "qAll")
            else:
                iAll = self.init_float([1, N], "iAll")
                iChemAll = self.init_float([1, N], "iChemAll")
                vAll = self.init_float([1, N], "vAll")
                kAll = self.init_float([1, N], "kAll")
                qAll = self.init_float([1, N], "qAll")

            with tf.name_scope('synaptic_connections'):
                # matrices with 1 where connection exists
                connE1E1, connE1E2, connE1I1, connE1I2, \
                connE2E1, connE2E2, connE2I1, connE2I2, \
                connI1E1, connI1E2, connI1I1, connI1I2, \
                connI2E1, connI2E2, connI2I1, connI2I2, \
                connIIg1, connIIg2, connIIgS, connDelete = self.makeConn(sG=self.sG, distrib='single_val')

                vectE1, vectE2, vectI1, vectI2 = self.makeVect()

                # mean synaptics weights
                if NE1 > 0:
                    wE1E1_init = self.wE1E1 / NE1
                    wE1I1_init = self.wE1I1 / (NI1 * NE1) ** 0.5
                    wI1E1_init = self.wI1E1 / (NI1 * NE1) ** 0.5
                else:
                    wE1E1_init, wE1I1_init, wI1E1_init = 0, 0, 0

                if NI1 > 0:
                    wI1I1_init = self.wI1I1 / NI1
                    g0 = self.g0 / NI1
                    g1 = self.g1 / NI1
                elif NI2 > 0:
                    g0 = self.g0 / NI2
                    g1, wI1I1_init = 0, 0
                else:
                    wI1I1_init, g0, g1 = 0, 0, 0

                if NE2 > 0:
                    wE1E2_init = self.wE1E2 / ((NE1 * NE2) ** 0.5)
                    wE2E1_init = self.wE2E1 / ((NE2 * NE1) ** 0.5)
                    wE2E2_init = self.wE2E2 / NE2
                    wE2I1_init = self.wE2I1 / ((NE2 * NI1) ** 0.5)
                    wI1E2_init = self.wI1E2 / ((NI1 * NE2) ** 0.5)

                    if NI2 > 0:
                        wE1I2_init = self.wE1I2 / (NI1 * NI2) ** 0.5
                        wE2I2_init = self.wE2I2 / (NI2 * NE2) ** 0.5
                        wI2E2_init = self.wI2E2 / (NI2 * NE2) ** 0.5
                    else:
                        wE1I2_init, wE2I2_init, wI2E2_init = 0, 0, 0
                else:
                    wE1E2_init, wE2E1_init, wE2E2_init, wE2I1_init, wE2I2_init, wE2I2_init, wI1E2_init, wI2E2_init = 0, 0, 0, 0, 0, 0, 0, 0

                if NI2 > 0:
                    if NE1 > 0:
                        wE1I2_init = self.wE1I2 / (NE1 * NI2) ** 0.5
                        wI2E1_init = self.wI2E1 / (NI2 * NE1) ** 0.5
                    else:
                        wE1I2_init, wI2E1_init = 0, 0
                    wI1I2_init = self.wI1I2 / (NI2 * NI1) ** 0.5
                    wI2I1_init = self.wI2I1 / (NI2 * NI1) ** 0.5
                    wI2I2_init = self.wI2I2 / NI2

                    g2 = self.g2 / NI2
                    gS = (g1 + g2) / 2
                else:
                    wE1I2_init, wE2I2_init, wI1I2_init, wI2E1_init, wI2E2_init, wI2I1_init, wI2I2_init = 0, 0, 0, 0, 0, 0, 0
                    g2 = 0
                    gS = 0

                WE1E1, WE1E2, WE1I1, WE1I2, \
                WE2E1, WE2E2, WE2I1, WE2I2, \
                WI1E1, WI1E2, WI1I1, WI1I2, \
                WI2E1, WI2E2, WI2I1, WI2I2, \
                Wgap1, Wgap2, WIIgS, _ = self.makeConn(
                    distrib=self.distrib, TF=True, mu=self.mu, sigma=self.sigma,
                    we1e1=wE1E1_init / dt, we1e2=wE1E2_init / dt, we1i1=wE1I1_init / dt, we1i2=wE1I2_init / dt,
                    we2e1=wE2E1_init / dt, we2e2=wE2E2_init / dt, we2i1=wE2I1_init / dt, we2i2=wE2I2_init / dt,
                    wi1e1=wI1E1_init / dt, wi1e2=wI1E2_init / dt, wi1i1=wI1I1_init / dt, wi1i2=wI1I2_init / dt,
                    wi2e1=wI2E1_init / dt, wi2e2=wI2E2_init / dt, wi2i1=wI2I1_init / dt, wi2i2=wI2I2_init / dt,
                    g1=g1, g2=g2, gS=gS
                )

                WII0 = WI1I1 + WI2I2

                WchemI = WI1E1 + WI1E2 + WI1I2 + WI2E1 + WI2E2 + WI2I1
                WchemE = WE1E1 + WE1E2 + WE1I1 + WE1I2 + WE2E1 + WE2E2 + WE2I1 + WE2I2
                Wchem = WchemI + WchemE

                wGap = tf.Variable(Wgap1 + Wgap2)
                # delete prop of GJs defined by self.propToDelete
                wGap = tf.multiply(wGap, connDelete)
                tf.global_variables_initializer().run()
                wGap = tf.Variable(wGap, name='wGap')

                # plasticity learning rates
                A_LTD = tf.constant(self.A_LTD, dtype=tf.float32)
                A_LTP = tf.constant(self.A_LTP, dtype=tf.float32)

            with tf.name_scope('membrane_var'):
                # Create variables for simulation state
                u = self.init_float([N, 1], 'u')
                b = self.init_float([N, 1], 'b')
                v = tf.Variable(
                    self.v0 * tf.random_normal([N, 1], mean=self.v_init_mean, stddev=self.v_init_std, name='v'))

                # currents
                iBack = self.init_float([N, 1], 'iBack')
                iChem = self.init_float([N, 1], 'iChem')
                input = tf.cast(tf.constant(self.input), tf.float32)

                tauvSubnet = tf.Variable(
                    self.tauv1 * vectI1 + self.tauv2 * vectI2 + (vectE1 + vectE2),
                    name="tauv")

            with tf.name_scope('simulation_params'):
                # stimulation
                kMult = tf.Variable(self.k, dtype=tf.float32)
                TImean = self.nu
                TEmean = self.nuE
                Nmean = TImean * (vectI1 + vectI2) + TEmean * (self.kNoiseE1 * vectE1 + self.kNoiseE2 * vectE2)
                # timestep
                dt = tf.constant(dt * 1.0, name="timestep")
                connectTime = self.connectTime
                stabTime = self.stabTime
                stopTime = self.stopTime
                # connection and plasticity times
                sim_index = tf.Variable(0.0, name="sim_index", dtype=tf.float32)
                one = tf.Variable(1.0)
                ones = tf.ones((1, N))

        #################################################################################
        ## Computation
        #################################################################################

        # Connect subnetworks
        with tf.name_scope('Connect'):
            g0_S = tf.reduce_mean(wGap * connIIg1) * ((NI1 + NE1) / NI1) ** 2 + \
                   tf.reduce_mean(wGap * connIIg2) * ((NI2 + NE2) / max(NI2, 1)) ** 2  # is 0 if NI2 == 0

            wGapS = g0_S * connIIgS
            connect = tf.group(
                wGap.assign(tf.add(wGap, wGapS))
            )

        # Currents
        with tf.name_scope('Currents'):
            WII = WII0 * (1 - 2 * kMult * wGap)

            iChem_ = iChem + dt / self.tau_I_I * (-iChem + tf.matmul(Wchem + WII, tf.to_float(vv), name="E/IPSPs"))

            # noisy input current
            iBack_ = iBack + dt / self.tau_I_I * (
                -iBack + tf.random_normal((N, 1), mean=0.0, stddev=1.0, dtype=tf.float32, name=None)) * (
                                 vectI1 + vectI2) + \
                     dt / self.tau_I_E * (-iBack + tf.random_normal((N, 1), mean=0.0, stddev=1.0, dtype=tf.float32,
                                                                    name=None)) * (vectE1 + self.kNoiseE2 * vectE2)
            # input_ = tf.gather(input, tf.to_int32(sim_index), axis=1)
            if self.input is not 0:
                input_ = tf.expand_dims(input[:, tf.to_int32(sim_index)], 1)
            else:
                input_ = input

            iEff_ = iBack_ * self.noiseScaling + input_ * (
                vectI1 + vectI2 + self.kInputE1 * vectE1 + self.kInputE2 * vectE2) + Nmean

            iGap_ = tf.matmul(wGap, v, name="GJ1") - tf.multiply(tf.reshape(tf.reduce_sum(wGap, 0), (N, 1)), v,
                                                                 name="GJ2")
            # sum all currents
            I_ = iGap_ + iChem_ + iEff_
        # Neuron models
        with tf.name_scope('Izhikevich'):

            '''
            IZH I + IAF E
            '''
            # voltage
            v_ = (v + dt / self.tau_v_I *
                  (0.25 * (v * v + 110 * v + 45 * 65) - u + I_)) * (vectI1 + vectI2) + \
                 (v + dt / self.tau_v_E * (-v + self.Rm * I_)) * (vectE1 + vectE2)

            # adaptation
            u_ = u + dt * 0.015 * (b * (v_ + 65) - u)

            # spikes
            vv_ = tf.to_float(tf.greater(v_, self.v_thresh_I)) * (vectI1 + vectI2) + \
                  tf.to_float(tf.greater(v_, self.v_thresh_E)) * (vectE1 + vectE2)

            # reset
            v_ = tf.multiply(vv_, self.v_r_I) * (vectI1 + vectI2) + tf.multiply(vv_, self.v_r_E) * (
                vectE1 + vectE2) + tf.multiply((1 - vv_),
                                               v_)

            u_ = u_ + self.u_a * vv_ * (vectI1 + vectI2)
            vvv = tf.to_float(v_ < -70)
            b_ = 10 * vvv + 2 * (1 - vvv)

        # Bursting
        with tf.name_scope('bursting'):
            LowSp_ = LowSp - dt / self.tau_burst * LowSp + vv_

        if self.spindles:
            if self.double_thresh:
                k_ = tf.to_float(tf.greater(LowSp_, self.burst_thresh))
            else:
                k_ = LowSp_
            q_ = q - dt / self.tau_q * (q - k_)
            p_ = tf.to_float(tf.greater(q_, self.q_thresh))
        else:
            p_ = tf.to_float(tf.greater(LowSp_, self.burst_thresh))

        # plasticity
        with tf.name_scope('plasticity'):
            A = tf.matmul(p_ * (vectI1 + vectI2), ones, name="bursts")  # bursts
            B = tf.matmul(vv_ * (1-p_) * (vectI1 + vectI2), ones, name="spikes")  # spikes

            dwLTD_ = A_LTD * tf.add(A, tf.transpose(A, name="tr_bursts"))
            if self.g0 == 0:
                # no bounds
                dwLTP_ = A_LTP * tf.add(B, tf.transpose(B, name="tr_spikes"))
            else:
                # LTP softbound
                dwLTP_ = A_LTP * (tf.multiply(tf.ones([N, N]) - wGap / g0, B + tf.transpose(B)))

            dwGap_ = tf.subtract(dwLTP_, dt * dwLTD_)

            # lower bound is 0
            wGap_ = wGap + dwGap_
            wGap_ = tf.clip_by_value(wGap_, clip_value_min=0, clip_value_max=np.inf)

            wGap_ = tf.multiply(wGap_, connDelete)
            wGap_before_ = tf.multiply(wGap_, connIIg1 + connIIg2)
            wGap_after_ = tf.multiply(wGap_, connIIg1 + connIIg2 + connIIgS)

        ##############################################################################################
        #
        # monitoring
        #
        ##############################################################################################
        with tf.name_scope('Debugging'):
            debug = tf.group(
                tf.scatter_update(Am, tf.to_int32(sim_index), tf.reduce_mean(A)),
                tf.scatter_update(Bm, tf.to_int32(sim_index), tf.reduce_mean(B)),
                tf.scatter_update(dwm, tf.to_int32(sim_index), tf.reduce_mean(dwGap_)))

        with tf.name_scope('Monitoring'):
            # PSTH
            vvmeanE1_ = tf.reduce_sum(vv_ * vectE1)
            # vvmeanE2_ = tf.reduce_sum(vv_ * vectE2)
            vvmeanI1_ = tf.reduce_sum(vv_ * vectI1)
            # vvmeanI2_ = tf.reduce_sum(vv_ * vectI2)

            # mean voltages
            vmeanE1_ = tf.reduce_sum(v_ * vectE1)
            # vmeanE2_ = tf.reduce_sum(v_ * vectE2)
            vmeanI1_ = tf.reduce_sum(v_ * vectI1)
            # vmeanI2_ = tf.reduce_sum(v_ * vectI2)

            # LFPs
            imeanE1_ = tf.reduce_sum(I_ * vectE1)
            # imeanE2_ = tf.reduce_sum(I_ * vectE2)
            imeanI1_ = tf.reduce_sum(I_ * vectI1)
            # imeanI2_ = tf.reduce_sum(I_ * vectI2)

            pmeanI1_ = tf.reduce_mean(p_ * vectI1)
            qmeanI1_ = tf.reduce_mean(q_ * vectI1)
            qqmeanI1_ = tf.reduce_mean(qq_ * vectI1)
            lowspmeanI1_ = tf.reduce_mean(LowSp_[:2])

            iGapm_ = tf.reduce_mean(iGap_ * vectI1)

            update = tf.group(
                tf.scatter_update(vvmE1, tf.to_int32(sim_index / monitor_step), vvmeanE1_),
                # tf.scatter_update(vvmE2, tf.to_int32(sim_index / monitor_step), vvmeanE2_),
                tf.scatter_update(vvmI1, tf.to_int32(sim_index / monitor_step), vvmeanI1_),
                # tf.scatter_update(vvmI2, tf.to_int32(sim_index / monitor_step), vvmeanI2_),

                tf.scatter_update(vmE1, tf.to_int32(sim_index / monitor_step), vmeanE1_),
                # tf.scatter_update(vmE2, tf.to_int32(sim_index / monitor_step), vmeanE2_),
                tf.scatter_update(vmI1, tf.to_int32(sim_index / monitor_step), vmeanI1_),
                # tf.scatter_update(vmI2, tf.to_int32(sim_index / monitor_step), vmeanI2_),

                tf.scatter_update(imE1, tf.to_int32(sim_index / monitor_step), imeanE1_),
                # tf.scatter_update(imE2, tf.to_int32(sim_index / monitor_step), imeanE2_),
                tf.scatter_update(imI1, tf.to_int32(sim_index / monitor_step), imeanI1_),
                # tf.scatter_update(imI2, tf.to_int32(sim_index / monitor_step), imeanI2_),

                tf.scatter_update(pmI1, tf.to_int32(sim_index / monitor_step), pmeanI1_),
                tf.scatter_update(qmI1, tf.to_int32(sim_index / monitor_step), qmeanI1_),
                tf.scatter_update(qqmI1, tf.to_int32(sim_index / monitor_step), qqmeanI1_),

                tf.scatter_update(lowspmI1, tf.to_int32(sim_index / monitor_step), lowspmeanI1_),

                tf.scatter_update(iGapm, tf.to_int32(sim_index / monitor_step), iGapm_),

            )
            update_single = tf.group(
                tf.scatter_update(vAll, tf.to_int32(sim_index), tf.reshape((v_), (N,))),
                tf.scatter_update(kAll, tf.to_int32(sim_index), tf.reshape((LowSp_), (N,))),
                tf.scatter_update(qAll, tf.to_int32(sim_index), tf.reshape((q_), (N,))),
                tf.scatter_update(iAll, tf.to_int32(sim_index), tf.reshape((I_), (N,))),
                tf.scatter_update(iChemAll, tf.to_int32(sim_index), tf.reshape((iChem_), (N,))),
            )

            update_sim_index = tf.group(
                sim_index.assign_add(one),
            )

        with tf.name_scope('Weights_monitoring'):
            WI1I1m_ = tf.reduce_sum(WII * connIIg1)
            g1m_ = tf.reduce_sum(wGap * connIIg1)
            # g2m_ = tf.reduce_sum(wGap * connIIg2)
            # gSm_ = tf.reduce_sum(wGap * connIIgS)
            update_weights = tf.group(
                tf.scatter_update(WI1I1m, tf.to_int32(sim_index / weight_step), WI1I1m_),
                tf.scatter_update(g1m, tf.to_int32(sim_index / weight_step), g1m_),
                # tf.scatter_update(g2m, tf.to_int32(sim_index / weight_step), g2m_),
                # tf.scatter_update(gSm, tf.to_int32(sim_index / weight_step), gSm_),
            )

        with tf.name_scope('Raster_Plot'):
            spike_update = tf.group(
                tf.scatter_update(spikes, tf.to_int32(sim_index), tf.reshape((vv_), (N,))),
            )

        # Operation to update the state
        step = tf.group(
            iChem.assign(iChem_),
            iBack.assign(iBack_),
            LowSp.assign(LowSp_),
            q.assign(q_),
            qq.assign(qq_),
            p.assign(p_),
            v.assign(v_),
            vv.assign(vv_),
            u.assign(u_),
            b.assign(b_)
        )

        # plasticity
        plast_before = tf.group(
            wGap.assign(wGap_before_),
        )
        plast_after = tf.group(
            wGap.assign(wGap_after_),
        )

        # initialize the graph
        tf.global_variables_initializer().run()

        ## chemical synapses
        # from E1
        self.WE1E1 = WE1E1.eval()
        self.WE1E2 = WE1E2.eval()
        self.WE1I1 = WE1I1.eval()
        self.WE1I2 = WE1I2.eval()

        # from E2
        self.WE2E1 = WE2E1.eval()
        self.WE2E2 = WE2E2.eval()
        self.WE2I1 = WE2I1.eval()
        self.WE2I2 = WE2I2.eval()

        # from I1
        self.WI1E1 = WI1E1.eval()
        self.WI1E2 = WI1E2.eval()
        self.WI1I1 = WI1I1.eval()
        self.WI1I2 = WI1I2.eval()
        self.connIIg1 = connIIg1.eval()
        self.connIIg2 = connIIg2.eval()

        # from I2
        self.WI2E1 = WI2E1.eval()
        self.WI2E2 = WI2E2.eval()
        self.WI2I1 = WI2I1.eval()
        self.WI2I2 = WI2I2.eval()

        self.WII = WII.eval()
        self.WII0 = WII0.eval()

        ## gap junctions connections
        self.connIIgS = connIIgS.eval()
        self.connIIg1 = connIIg1.eval()
        self.connIIg2 = connIIg2.eval()
        self.wGap0 = wGap.eval()

        ops = {'before': [step, plast_before],
               'after': [step, plast_after],
               'static': [step]
               }

        if self.monitor_single:
            update = [update, update_single]

        if monitor_step == 1:
            for k, v in ops.items():
                ops[k] = v + [update]

        if weight_step == 1:
            for k, v in ops.items():
                ops[k] = v + [update_weights]

        if self.spikeMonitor:
            for k, v in ops.items():
                ops[k] = v + [spike_update]

        if self.debug:
            for k, v in ops.items():
                ops[k] = v + [debug]

        t0 = time.time()
        for i in range(T): # use trange to show progress
            # Step simulation
            if i == connectTime:
                self.sess.run([connect])

            if i < stabTime or i > stopTime:
                self.sess.run(ops['static'],
                              options=self.run_options,
                              run_metadata=self.run_metadata
                              )
            else:
                self.sess.run(ops['after'],
                              options=self.run_options,
                              run_metadata=self.run_metadata
                              )

            if monitor_step != 1 and i % monitor_step == 0:
                self.sess.run([update])

            if weight_step != 1 and i % weight_step == 0:
                self.sess.run([update_weights])

            self.sess.run([update_sim_index])

        # debugging
        self.Am = Am.eval()
        self.Bm = Bm.eval()
        self.dwm = dwm.eval()

        # monitoring variables
        self.wGapE = wGap.eval()
        self.vvmE1 = vvmE1.eval()
        # self.vvmE2 = vvmE2.eval()
        self.vvmI1 = vvmI1.eval()
        # self.vvmI2 = vvmI2.eval()

        self.vmE1 = vmE1.eval()
        # self.vmE2 = vmE2.eval()
        self.vmI1 = vmI1.eval()
        # self.vmI2 = vmI2.eval()

        self.imI1 = imI1.eval()
        # self.imI2 = imI2.eval()
        self.imE1 = imE1.eval()
        # self.imE2 = imE2.eval()

        self.iAll = iAll.eval().T
        self.iChemAll = iChemAll.eval().T
        self.vAll = vAll.eval().T
        self.kAll = kAll.eval().T
        self.qAll = qAll.eval().T

        self.pmI1 = pmI1.eval()
        self.qmI1 = qmI1.eval()
        self.qqmI1 = qqmI1.eval()
        self.lowspmI1 = lowspmI1.eval()

        self.WI1I1m = WI1I1m.eval()
        self.WIIe = WII.eval()
        self.gm1 = g1m.eval()
        # self.gm2 = g2m.eval()
        # self.gmS = gSm.eval()
        self.iGapm = iGapm.eval()
        self.burstingActivity1 = np.mean(self.pmI1)
        self.spikingActivity1 = np.mean(self.vvmI1)
        self.connDelete = connDelete.eval()
        if self.spikeMonitor:
            self.raster = spikes.eval()

        # print simulation duration
        print('\n Duration: %.2f\n' % (time.time() - t0))

        self.sess.close()
