import numpy as np
from numpy.linalg import norm


# This has the same general interface as tf.Metric, but it doesn't
# actually implement tf.Metric because we may need to perform
# computations involving Python logic.
class PCKMultiPerson:
    def __init__(self, threshold=0.2, postprocess_func=None, name=None):
        self.threshold = threshold
        self.postprocess_func = postprocess_func
        self.name = "pck_multi_person" if (name is None) else name
        self.n_total = 0
        self.n_correct = 0

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {"threshold": self.threshold, "postprocess_func": self.postprocess_func}

    def reset_states(self):
        self.n_total = 0
        self.n_correct = 0

    def result(self):
        return 0.0 if (self.n_total == 0) else self.n_correct / self.n_total

    def update_state(self, y_true, y_pred):
        if self.postprocess_func is not None:
            people_true_batch, people_pred_batch = self.postprocess_func(y_true, y_pred)
        else:
            people_true_batch = y_true
            people_pred_batch = y_pred

        for i, people_true in enumerate(people_true_batch):
            people_pred = _match_people(people_true, people_pred_batch[i])
            for person_true, person_pred in zip(people_true, people_pred):
                n_joints, n_matching = _count_pck_joints(person_true, person_pred, self.threshold)
                self.n_total += n_joints
                self.n_correct += n_matching


class PCKhMultiPerson:
    def __init__(self, threshold=0.5, head_scale=0.6, postprocess_func=None, name=None):
        self.threshold = threshold
        self.head_scale = head_scale
        self.postprocess_func = postprocess_func
        self.name = "pckh_multi_person" if (name is None) else name
        self.n_total = 0
        self.n_correct = 0

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {
            "threshold": self.threshold,
            "head_scale": self.head_scale,
            "true_postprocess_func": self.postprocess_func,
        }

    def reset_states(self):
        self.n_total = 0
        self.n_correct = 0

    def result(self):
        return 0.0 if (self.n_total == 0) else self.n_correct / self.n_total

    def update_state(self, y_true, y_pred):
        if self.postprocess_func is not None:
            people_true_batch, people_pred_batch, heads_batch = self.postprocess_func(
                y_true, y_pred
            )
        else:
            people_pred_batch = y_pred
            people_true_batch, heads_batch = y_true

        for i, people_true in enumerate(people_true_batch):
            people_pred = _match_people(people_true, people_pred_batch[i])

            # Count the number of ground-truth joints where the
            # predicted joint is closer than
            # threshold * head_scale * head_size.
            for j, (person_true, person_pred) in enumerate(zip(people_true, people_pred)):
                x_1, x_2, y_1, y_2 = heads_batch[i, j]
                head_size = norm(np.array([x_1, y_1]) - np.array([x_2, y_2]))
                for joint_true, joint_pred in zip(person_true, person_pred):
                    if np.any(np.isnan(joint_true)):
                        continue
                    self.n_total += 1
                    if np.any(joint_pred < 0):
                        continue
                    error = norm(joint_true - joint_pred)
                    if error < self.threshold * self.head_scale * head_size:
                        self.n_correct += 1


class PCKSinglePerson:
    def __init__(self, threshold=0.2, postprocess_func=None, name=None):
        self.threshold = threshold
        self.postprocess_func = postprocess_func
        self.name = "pck_single_person" if (name is None) else name
        self.n_total = 0
        self.n_correct = 0

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {"threshold": self.threshold, "postprocess_func": self.postprocess_func}

    def reset_states(self):
        self.n_total = 0
        self.n_correct = 0

    def result(self):
        return 0.0 if (self.n_total == 0) else self.n_correct / self.n_total

    def update_state(self, y_true, y_pred):
        if self.postprocess_func is not None:
            people_true_batch, people_pred_batch = self.postprocess_func(y_true, y_pred)
        else:
            people_true_batch = y_true
            people_pred_batch = y_pred

        for person_true, person_pred in zip(people_true_batch, people_pred_batch):
            n_joints, n_matching = _count_pck_joints(person_true, person_pred, self.threshold)
            self.n_total += n_joints
            self.n_correct += n_matching


def _count_pck_joints(person_true, person_pred, threshold):
    # Calculate the height and width of the person's bounding box.
    x = person_true[:, 0]
    x = x[~np.isnan(x)]
    y = person_true[:, 1]
    y = y[~np.isnan(y)]
    h = np.max(y) - np.min(y)
    w = np.max(x) - np.min(x)

    # Count the number of ground-truth joints where the
    # predicted joint is closer than threshold * max(h, w).
    n_joints = 0
    n_matching = 0
    for joint_true, joint_pred in zip(person_true, person_pred):
        if np.any(np.isnan(joint_true)):
            continue
        else:
            n_joints += 1
        if np.any(joint_pred < 0):
            continue
        elif norm(joint_true - joint_pred) < threshold * np.maximum(h, w):
            n_matching += 1
    return n_joints, n_matching


# Greedily matches people by the mean distance between their joints.
def _match_people(people_true, people_pred_unmatched):
    people_pred = []
    for person_true in people_true:
        if len(people_pred_unmatched) == 0:
            people_pred.append(np.full_like(person_true, -1.0))
            continue

        j_best = 0
        mean_error_best = float("inf")
        for j, person_pred in enumerate(people_pred_unmatched):
            total_error = 0.0
            n_joints = 0
            for joint_true, joint_pred in zip(person_true, person_pred):
                if np.any(np.isnan(joint_true)) or np.any(joint_pred < 0):
                    continue
                total_error += norm(joint_true - joint_pred)
                n_joints += 1
            mean_error = (total_error / n_joints) if (n_joints > 0) else float("inf")
            if mean_error < mean_error_best:
                j_best = j
                mean_error_best = mean_error
        people_pred.append(people_pred_unmatched.pop(j_best))

    return people_pred
