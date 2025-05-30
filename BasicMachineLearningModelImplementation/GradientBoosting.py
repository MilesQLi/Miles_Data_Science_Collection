import torch
import numpy as np 
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingBinaryClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, reg_lambda=1.0):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.trees = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        y = y.astype(np.float64)
        self.F0 = np.log(y.mean() / (1 - y.mean()))
        F = np.full_like(y, self.F0)
        for _ in range(self.n_estimators):
            p = self._sigmoid(F)
            grad = y - p
            hess = p * (1 - p) + self.reg_lambda
            target = grad / hess
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, target)
            update = tree.predict(X)
            F += self.lr * update
            self.trees.append(tree)

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.F0)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        proba = self._sigmoid(F)
        return np.vstack([1 - proba, proba]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)



##################################################################

class TreeNode:
    def __init__(self, is_leaf=False, leaf_value=None, feature_idx=None, threshold=None,
                 left_child=None, right_child=None):
        self.is_leaf = is_leaf
        self.leaf_value = leaf_value  # Output value if it's a leaf
        self.feature_idx = feature_idx # Index of feature to split on
        self.threshold = threshold     # Threshold for the split
        self.left_child = left_child
        self.right_child = right_child

class PyTorchGBClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, reg_lambda=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda # L2 regularization term
        self.trees_ = []
        self.initial_prediction_ = None
        if random_state is not None:
            torch.manual_seed(random_state) # For reproducibility of feature selection if any randomness was added

    def _calculate_leaf_value(self, gradients, hessians):
        # Optimal leaf value calculation
        return -torch.sum(gradients) / (torch.sum(hessians) + self.reg_lambda)

    def _calculate_split_gain(self, gradients, hessians, left_indices, right_indices):
        # Gain = 0.5 * [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda) ]
        # The 0.5 factor can be ignored as we are only comparing gains
        
        g_left, h_left = gradients[left_indices], hessians[left_indices]
        g_right, h_right = gradients[right_indices], hessians[right_indices]

        sum_g_left, sum_h_left = torch.sum(g_left), torch.sum(h_left)
        sum_g_right, sum_h_right = torch.sum(g_right), torch.sum(h_right)
        
        # If a child node is empty or has near-zero hessian sum, its contribution to gain is 0 or negative
        # This check prevents division by zero or instability
        gain_left = (sum_g_left**2) / (sum_h_left + self.reg_lambda) if (sum_h_left + self.reg_lambda) > 1e-6 else 0
        gain_right = (sum_g_right**2) / (sum_h_right + self.reg_lambda) if (sum_h_right + self.reg_lambda) > 1e-6 else 0
        
        sum_g_total = sum_g_left + sum_g_right
        sum_h_total = sum_h_left + sum_h_right
        gain_total = (sum_g_total**2) / (sum_h_total + self.reg_lambda) if (sum_h_total + self.reg_lambda) > 1e-6 else 0
        
        gain = gain_left + gain_right - gain_total
        return gain

    def _find_best_split(self, X_subset, gradients_subset, hessians_subset):
        n_samples, n_features = X_subset.shape
        best_gain = -float('inf')
        best_feature_idx = None
        best_threshold = None

        if n_samples < self.min_samples_split:
            return None, None, None

        for feature_idx in range(n_features):
            unique_values = torch.unique(X_subset[:, feature_idx])
            if len(unique_values) <= 1: # Cannot split if all values are the same
                continue

            # Consider midpoints between unique sorted values as potential thresholds
            # For simplicity, using unique values themselves.
            # A more robust approach might sort and use midpoints.
            thresholds = unique_values 
            if len(thresholds) > 10: # Limit number of thresholds to check for speed (crude subsampling)
                 # Pick some thresholds if too many, can be random or percentiles
                indices = torch.randperm(len(thresholds))[:10]
                thresholds = thresholds[indices]


            for threshold in thresholds:
                left_indices = torch.where(X_subset[:, feature_idx] <= threshold)[0]
                right_indices = torch.where(X_subset[:, feature_idx] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue # Split doesn't separate samples

                gain = self._calculate_split_gain(gradients_subset, hessians_subset, left_indices, right_indices)

                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        if best_gain <= 0: # No positive gain found
            return None, None, None
            
        return best_feature_idx, best_threshold, best_gain

    def _build_tree(self, X, gradients, hessians, current_depth):
        if current_depth >= self.max_depth or len(X) < self.min_samples_split:
            leaf_value = self._calculate_leaf_value(gradients, hessians)
            return TreeNode(is_leaf=True, leaf_value=leaf_value)

        best_feature_idx, best_threshold, best_gain = self._find_best_split(X, gradients, hessians)

        if best_feature_idx is None: # No beneficial split found
            leaf_value = self._calculate_leaf_value(gradients, hessians)
            return TreeNode(is_leaf=True, leaf_value=leaf_value)

        # Split data
        left_mask = X[:, best_feature_idx] <= best_threshold
        right_mask = X[:, best_feature_idx] > best_threshold
        
        # Ensure children are non-empty if split happened
        if not torch.any(left_mask) or not torch.any(right_mask):
            leaf_value = self._calculate_leaf_value(gradients, hessians)
            return TreeNode(is_leaf=True, leaf_value=leaf_value)

        left_child = self._build_tree(X[left_mask], gradients[left_mask], hessians[left_mask], current_depth + 1)
        right_child = self._build_tree(X[right_mask], gradients[right_mask], hessians[right_mask], current_depth + 1)

        return TreeNode(feature_idx=best_feature_idx, threshold=best_threshold,
                        left_child=left_child, right_child=right_child)

    def fit(self, X, y):
        if isinstance(X, np.ndarray): X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray): y = torch.from_numpy(y).float()
        
        n_samples, _ = X.shape

        # Initial prediction: log-odds of the mean of y
        # Add epsilon to prevent log(0) or division by zero
        p_avg = torch.mean(y.float())
        self.initial_prediction_ = torch.log((p_avg + 1e-9) / (1.0 - p_avg + 1e-9))
        
        # Current accumulated predictions (scores)
        current_scores = torch.full((n_samples,), self.initial_prediction_.item(), dtype=torch.float32)

        self.trees_ = []
        for i in range(self.n_estimators):
            # Probabilities from current scores
            probabilities = torch.sigmoid(current_scores)

            # Gradients and Hessians for binary log-loss
            # Gradient g = (probability - true_label)
            # Hessian h = probability * (1 - probability)
            gradients = probabilities - y
            hessians = probabilities * (1 - probabilities)
            
            # Clamp Hessians to avoid very small values which can lead to instability
            hessians = torch.clamp(hessians, min=1e-6)


            tree = self._build_tree(X, gradients, hessians, current_depth=0)
            self.trees_.append(tree)

            # Update scores with the new tree's predictions (leaf values)
            # We need to get predictions for each sample from the new tree
            tree_predictions = self._predict_tree_values(tree, X)
            current_scores += self.learning_rate * tree_predictions
            
            # print(f"Tree {i+1}/{self.n_estimators} built. Example score: {current_scores[0].item()}")


    def _traverse_tree(self, node, x_sample):
        if node.is_leaf:
            return node.leaf_value
        if x_sample[node.feature_idx] <= node.threshold:
            return self._traverse_tree(node.left_child, x_sample)
        else:
            return self._traverse_tree(node.right_child, x_sample)

    def _predict_tree_values(self, tree, X):
        predictions = torch.zeros(len(X), dtype=torch.float32)
        for i in range(len(X)):
            predictions[i] = self._traverse_tree(tree, X[i])
        return predictions

    def predict_proba(self, X):
        if isinstance(X, np.ndarray): X = torch.from_numpy(X).float()
        n_samples = X.shape[0]

        current_scores = torch.full((n_samples,), self.initial_prediction_.item(), dtype=torch.float32)

        for tree in self.trees_:
            tree_predictions = self._predict_tree_values(tree, X)
            current_scores += self.learning_rate * tree_predictions
        
        # Probabilities for class 1
        proba_class1 = torch.sigmoid(current_scores)
        # Probabilities for class 0
        proba_class0 = 1.0 - proba_class1
        
        # Return in scikit-learn format [P(class 0), P(class 1)]
        return torch.stack([proba_class0, proba_class1], dim=1)

    def predict(self, X):
        probas = self.predict_proba(X)
        # Predict class 1 if P(class 1) > 0.5
        return (probas[:, 1] > 0.5).int()
