import numpy as np
from scipy.signal import savgol_filter as sf
import scipy.cluster as scl

class HeuristicAnalyser:
    """
    Runs heuristic analysis on a parameter sets and returns the set either unaltered, 
    or modified and bounds-checked.
    """
    def __init__(self, params, costs, bounds, bump_incrementer, drift_settings, initial_count):

        self.params = params
        self.costs = costs
        self.bounds = bounds

        self.bump_incrementer = bump_incrementer
        self.initial_count = initial_count

        self.num_params = len(bounds)
        if len(costs) > 0:
            best_params_ind = np.argmin(costs)
            self.best_params = params[best_params_ind]
        else:
            self.best_params = []

        self.drift_compensation = drift_settings[0]
        self.drift_limit = drift_settings[1]
        self.drift_memory = drift_settings[2]

        self.runs_without_increase = 0
        if len(costs) > 0:
            current_cost = costs[-1]
            for i in range(len(costs) - 2, -1, -1):
                last_cost = costs[i]
                if last_cost >= current_cost:
                    self.runs_without_increase += 1
                else:
                    break
                current_cost = last_cost

    def analyse(self, new_params):
        
        # scaling function for distances
        mins, maxs = list(zip(*self.bounds))
        scale_func = lambda X: (np.array(X) - np.array(mins)) / (np.array(maxs) - np.array(mins))

        param_span = (np.array(maxs) - np.array(mins))
        bump_scale = lambda x: x
        bump_start = 0.001
        last_params = np.array([np.inf] * self.num_params)

        bumped = False

        # determine the distance and whether we should bump
        if len(self.best_params) > 0:
            distance = np.sum(np.square(scale_func(self.best_params) - scale_func(new_params))) / self.num_params
            distance_last = np.sum(np.square(scale_func(last_params) - scale_func(new_params))) / self.num_params

            if self.runs_without_increase > 3 and (distance < 0.03 or distance_last < 0.05):

                bumped = True

                # increment the bump if we need to
                self.runs_without_increase = 0
                if self.bump_incrementer > 5:
                    self.bump_incrementer = 0

                    if bump_scale(bump_start) >= 0.12:
                        bump_start = 0.001

                    bump_start = bump_start + bump_start * 2
                else:
                    self.bump_incrementer += 1

                offset = param_span * np.random.uniform(-0.5, 0.5, 1)

                if self.bump_incrementer % 2 == 0:
                    polarity_mask = np.sign(
                        (new_params - np.array(mins)) + (new_params - np.array(maxs))) * -1

                    new_params = new_params + np.multiply(polarity_mask, param_span * bump_scale(bump_start)) + offset

                    new_params = new_params.tolist()

                elif self.bump_incrementer % 3 == 0:
                    p = np.array(self.params)
                    c = np.array(self.costs)

                    c_sorted = sorted(self.costs)
                    cmax = c_sorted[int(len(c) * 0.3)]
                    mask = np.where(c < cmax)

                    try:
                        clusters = scl.hierarchy.fclusterdata(p[mask], 0.75)
                        cc = zip(clusters, c[mask], p[mask])
                        cc = sorted(cc, key=lambda x: x[1])

                        unique_rows = []
                        current_cluster = -1
                        for row in cc:
                            if row[0] != current_cluster:
                                current_cluster = row[0]
                                unique_rows.append(row)

                        unique_clusters, unique_costs, unique_params = zip(*unique_rows)

                        bump_dir = np.sign(
                            unique_params[0] - unique_params[np.random.randint(0, len(unique_rows), 1)[0]])

                        new_params = self.best_params + np.multiply(bump_dir, param_span * bump_scale(bump_start))

                        new_params = new_params.tolist()

                        print('Clustered successfully!')

                    except ValueError:
                        print('Failed to cluster...')
                        print(p[mask])

                else:
                    if not self.drift_compensation:
                        print('Bumping with filter ...' + '-' * 20)
                        bump_params = []
                        for j in range(self.num_params):
                            xs = []
                            ys = []
                            for ct, pt in zip(self.costs, self.params):
                                xs.append(pt[j])
                                ys.append(ct)

                            zipped = sorted(zip(xs, ys), key=lambda x: x[0])
                            xs, ys = zip(*zipped)

                            avg = 41
                            if len(ys) < 41:
                                avg = int(len(ys) / 2)

                            if avg % 2 == 0:
                                avg += 1

                            polyorder = min(10, int(avg) - 1)
                            yfil = sf(ys, int(avg), polyorder)

                            best_val = xs[np.argmin(yfil)]
                            new_val = self.best_params[j] + (np.sign(best_val - self.best_params[j]) * (best_val - self.best_params[j]) * bump_scale(bump_start))
                            bump_params.append(new_val)

                        new_params = bump_params

                    else:
                        print('Bumping with drift ...' + '-' * 20)
                        counter = 0
                        min_cost = [np.inf, -1]
                        final_val = 0
                        while True:
                            i = self.initial_count + (counter * self.drift_limit)
                            j = self.initial_count + (counter * self.drift_limit) + self.drift_memory

                            if j > len(self.complete_training_set[0]):
                                break

                            min_val = np.argmin(self.complete_training_set[1][i:j])
                            if self.complete_training_set[1][i + min_val] < min_cost[0]:
                                min_cost = [self.complete_training_set[1][i + min_val], min_val + i]

                            final_val = i + min_val
                            counter += 1

                        start = np.array(self.complete_training_set[0][min_cost[1]])
                        end = np.array(self.complete_training_set[0][final_val])
                        mask = np.sign(end - start)
                        new_params = end + mask * bump_scale(bump_start)

# ==============================================================================================================================================
# ================================================== CONFORM PARAMS TO USER-SET BOUNDS =========================================================
# ==============================================================================================================================================

                        # restrict to the bounds
                        for i in range(self.num_params):
                            if new_params[i] > self.bounds[i][1]:
                                new_params[i] = self.bounds[i][1]

                            if new_params[i] < self.bounds[i][0]:
                                new_params[i] = self.bounds[i][0]

        return new_params, bumped