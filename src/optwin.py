from capymoa.drift.base_detector import BaseDriftDetector
import numpy as np
from scipy.stats import t as t_stat
from scipy.optimize import fsolve
import scipy.stats
import math
import warnings
from tqdm import tqdm


class OPTWIN(BaseDriftDetector):
    class Circular_list():
        def __init__(self, maxSize):
            self.W = [0 for i in range(maxSize)]
            self.length = 0
            self.init = 0

        def pos(self, idx):
            if self.init + idx < len(self.W):
                return self.init+idx
            else:
                return self.init + idx - len(self.W)

        def add(self, x):
            position = self.pos(self.length)
            self.length = self.length +1
            self.W[position] = x

        def pop_first(self):
            x = self.W[self.init]
            position = self.pos(1)
            self.init = position
            self.length = self.length-1
            return x

        def get(self, idx):
            position = self.pos(idx)
            return self.W[position]

        def get_interval(self, idx1, idx2):
            position1 = self.pos(idx1)
            position2 = self.pos(idx2)
            if position1 <= position2:
                return self.W[position1:position2]
            else:
                return self.W[position1:]+self.W[:position2]
    
    def __init__(self, confidence_final=0.99, rigor=0.5, empty_w=True, w_length_max=25000, w_length_min=30, minimum_noise=1e-6):
        """Initialize the OPTWIN drift detector.

        Args:
            confidence_final (float): Confidence value chosen by user (default: 0.999)
            rigor (float): Rigorousness of drift identification (default: 0.5)
            empty_w (bool): Empty window when drift is detected (default: True)
            w_length_max (int): Maximum window size (default: 50000)
            w_length_min (int): Minimum window size (default: 30)
            minimum_noise (float): Noise to be added to stdev in case it is 0 (default: 1e-6)
        """
        super().__init__()
        warnings.filterwarnings('ignore', message='The iteration is not making good progress')
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        
        # OPTWIN parameters
        self.confidence_final = confidence_final
        self.rigor = rigor
        self.w_length_max = w_length_max
        self.w_length_min = w_length_min
        self.minimum_noise = minimum_noise
        self.pre_compute_optimal_cut = True
        self.empty_w = empty_w
        
        self.W = self.Circular_list(w_length_max)
        self.opt_cut = []
        self.opt_phi = []
        self.t_stats = []
        self.t_stats_warning = []
        self.last_opt_cut = 0
        self.drift_type = []
        self.confidence = pow(self.confidence_final, 1/4)
        self.confidence_warning = 0.98
        
        self.t_score = lambda n : t_stat.ppf(self.confidence, df=self.degree_freedom(n))
        self.t_score_warning = lambda n : t_stat.ppf(self.confidence_warning, df=self.degree_freedom(n))
        self.f_test = lambda n : scipy.stats.f.ppf(q=self.confidence, dfn=(n*self.W.length)-1, dfd=self.W.length-(n*self.W.length)-1)
        self.degree_freedom = lambda n : pow(((1/max(self.W.length*n,1e-15))+((1/pow(self.f_test(n),2))/((1-n)*self.W.length))),2)/((1/max((pow((self.W.length*n),2)*((self.W.length*n)-1)),1e-15))+(pow((1/pow(self.f_test(n),2)),2)/max((pow(((1-n)*self.W.length),2)*(((1-n)*self.W.length)-1)),1e-15)))
        self.t_test = lambda n : self.rigor - (self.t_score(n) * np.sqrt((1/(self.W.length*n))+((1* self.f_test(n))/((1-n)*self.W.length))))
        
        # Running stdev and avg
        self.stdev_new = 0
        self.summation_new = 0
        self.count_new = 0
        self.S_new = 0
        self.stdev_h = 0
        self.summation_h = 0
        self.count_h = 0
        self.S_h = 0
        
        self.in_concept_change = False
        self.in_warning_zone = False
        self.estimation = 0.0
        self.delay = 0.0
        self.sequence_drifts = 0
        self.sequence_no_drifts = 0
        
        # Pre-compute optimal cut for all possible window sizes
        if self.pre_compute_optimal_cut:
            self.opt_cut, self.opt_phi, self.t_stats, self.t_stats_warning = self.pre_compute_cuts(self.opt_cut, self.opt_phi, self.t_stats, self.t_stats_warning)
            
        if len(self.opt_cut) == 0:
            self.opt_cut = [0 for i in range(w_length_min)]
            self.opt_phi = [0 for i in range(w_length_min)]
            self.t_stats = [0.0 for i in range(w_length_min)]
            self.t_stats_warning = [0.0 for i in range(w_length_min)]
            
        if len(self.opt_cut) >= w_length_max and len(self.opt_phi) >= w_length_max:
            self.pre_compute_optimal_cut = True

    def pre_compute_cuts(self, opt_cut, opt_phi, t_stats, t_stats_warning):
        if len(opt_cut) != 0 and len(opt_phi) != 0 and len(t_stats) != 0 and len(t_stats_warning) != 0:
            return opt_cut, opt_phi, t_stats, t_stats_warning
        self.W = self.Circular_list(self.w_length_max)

        pbar = tqdm(range(self.w_length_max + 1), desc="Pre-computing optimal cuts")

    
        for i in pbar:
            if i < self.w_length_min:
                opt_cut.append(0)
                opt_phi.append(0)
                t_stats.append(0.0)
                t_stats_warning.append(0.0)
            else:
                optimal_cut = fsolve(self.t_test, (self.W.length-30)/self.W.length)
                
                tolerance = 1e-6
                if abs(self.t_test(optimal_cut[0])) <= tolerance:
                    optimal_cut = math.floor(optimal_cut[0]*self.W.length)
                else:
                    optimal_cut = math.floor((self.W.length/2)+1)
                
                phi_opt = scipy.stats.f.ppf(q=self.confidence, dfn=optimal_cut-1, dfd=self.W.length-optimal_cut-1) 
                opt_cut.append(optimal_cut)
                opt_phi.append(phi_opt)
                t_stats.append(self.t_score(optimal_cut/i))
                t_stats_warning.append(self.t_score_warning(optimal_cut/i))
            self.W.add(1)
        
        self.W = self.Circular_list(self.w_length_max)
        return opt_cut, opt_phi, t_stats, t_stats_warning
    
    def insert_to_W(self, x):
        self.W.add(x)
        self.stdev_new, self.summation_new, self.count_new, self.S_new = self.add_running_stdev(self.summation_new, self.count_new, self.S_new, [x])
        
        # Check if window is too big
        if self.W.length > self.w_length_max:
            pop = self.W.pop_first()
            self.stdev_h, self.summation_h, self.count_h, self.S_h = self.pop_from_running_stdev(self.summation_h, self.count_h, self.S_h, [pop])
            self.stdev_new, self.summation_new, self.count_new, self.S_new = self.pop_from_running_stdev(self.summation_new, self.count_new, self.S_new, [self.W.get(self.last_opt_cut)])
            self.stdev_h, self.summation_h, self.count_h, self.S_h = self.add_running_stdev(self.summation_h, self.count_h, self.S_h, [self.W.get(self.last_opt_cut)])
        return
    
    def add_running_stdev(self, summation, count, S, x):
        summation += sum(x)
        count += len(x)
        S += sum([i*i for i in x])
                
        if (count > 1 and S > 0):
            stdev = math.sqrt((count*S) - (summation*summation)) / count
            return stdev, summation, count, S
        else:
            return 0, summation, count, S
        
    def pop_from_running_stdev(self,summation, count, S, x):
        summation -= sum(x)
        count -= len(x)
        S -= sum([i*i for i in x])
        
        if (count > 1 and S > 0):
            stdev = math.sqrt((count*S) - (summation*summation)) / count
            return stdev, summation, count, S
        else:
            return 0, summation, count, S 
    

    def add_element(self, x) -> None:
        # Add new element to window
        self.idx += 1
        self.data.append(x)
        self.insert_to_W(x)
        self.delay = 0
        
        # Check if window is too small
        if self.W.length < self.w_length_min:
            self.in_concept_change = False
            self.in_warning_zone = False
            return
  
        #check optimal window cut and phi
        #get pre-calculated optimal window cut and phi
        optimal_cut = self.opt_cut[self.W.length]
        phi_opt = self.opt_phi[self.W.length]
                     
        # Update running stdev and avg
        if optimal_cut > self.last_opt_cut: # Remove elements from window_new and add them to window_h
            self.stdev_new, self.summation_new, self.count_new, self.S_new = self.pop_from_running_stdev(self.summation_new, self.count_new, self.S_new, self.W.get_interval(self.last_opt_cut,optimal_cut))
            self.stdev_h, self.summation_h, self.count_h, self.S_h = self.add_running_stdev(self.summation_h, self.count_h, self.S_h, self.W.get_interval(self.last_opt_cut,optimal_cut))
        elif optimal_cut < self.last_opt_cut: # Remove elements from window_h and add them to window_new
            self.stdev_h, self.summation_h, self.count_h, self.S_h = self.pop_from_running_stdev(self.summation_h, self.count_h, self.S_h, self.W.get_interval(optimal_cut,self.last_opt_cut))
            self.stdev_new, self.summation_new, self.count_new, self.S_new = self.add_running_stdev(self.summation_new, self.count_new, self.S_new, self.W.get_interval(optimal_cut,self.last_opt_cut))
    
        avg_h = self.summation_h / self.count_h
        avg_new = self.summation_new / self.count_new

        stdev_h = math.sqrt((self.count_h*self.S_h) - (self.summation_h*self.summation_h)) / self.count_h
        stdev_new = math.sqrt((self.count_new*self.S_new) - (self.summation_new*self.summation_new)) / self.count_new
        
        self.last_opt_cut = optimal_cut

        
        # Add minimal noise to stdev
        stdev_h += self.minimum_noise
        stdev_new += self.minimum_noise
        
        if self.pre_compute_optimal_cut:
            t_stat = self.t_stats[self.W.length]
            t_stat_warning = self.t_stats_warning[self.W.length]
        else:
            t_stat = self.t_score(optimal_cut/self.W.length)
            t_stat_warning = self.t_score_warning(optimal_cut/self.W.length)
        
        
        # t-test
        t_test_result = (avg_new-avg_h) / (math.sqrt((stdev_new/(self.W.length-optimal_cut))+(stdev_h/optimal_cut)))
        if  t_test_result > t_stat:
            self.in_concept_change = True
            self.in_warning_zone = False
        elif t_test_result > t_stat_warning:
            self.in_warning_zone = True
            self.in_concept_change = False
        else:
            self.in_warning_zone = False
            self.in_concept_change = False
    
        # f-test
        if (stdev_new*stdev_new/(stdev_h*stdev_h)) > phi_opt:
        
            if avg_h - avg_new < 0:
                self.in_concept_change = True
            else:
                self.empty_window()
                self.in_concept_change = False
                self.in_warning_zone = False
        else:
            self.in_concept_change = False
            self.in_warning_zone = False
        
        if self.in_warning_zone:
            self.warning_index.append(self.idx)

        if self.in_concept_change:
            self.detection_index.append(self.idx)

    
    def empty_window(self):
        self.W = self.Circular_list(self.w_length_max)
        self.stdev_new = 0
        self.summation_new = 0
        self.count_new = 0
        self.S_new = 0
        self.stdev_h = 0
        self.summation_h = 0
        self.count_h = 0
        self.S_h = 0
        self.last_opt_cut = 0
            
    
    def get_length_estimation(self):
        self.estimation = self.W.length
        return self.W.length
    


    def get_params(self):
        """Get the current settings of the detector.
        
        Returns:
            dict: Dictionary containing the current settings
        """
        return {
            'confidence_final': self.confidence_final,
            'rigor': self.rigor,
            'w_length_max': self.w_length_max,
            'w_length_min': self.w_length_min
        }
            