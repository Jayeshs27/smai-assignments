from common import *

class MultiLabelMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def accuracy_score(self):
        correct_labels = np.all(self.y_true == self.y_pred, axis=1)
        return np.mean(correct_labels)

    def precision_score(self, method='macro'):
        tp_per_label = np.sum((self.y_true == 1) & (self.y_pred == 1), axis=0)
        pred_sum_per_label = np.sum(self.y_pred, axis=0)
        precision_per_label = np.divide(tp_per_label, pred_sum_per_label, 
                                    out=np.zeros_like(tp_per_label, dtype=float), 
                                    where=pred_sum_per_label != 0)
        
        if method == 'macro':
            return np.mean(precision_per_label)
        elif method == 'micro':
            tp = np.sum((self.y_true == 0) & (self.y_pred == 1))  
            fp = np.sum(self.y_pred) - tp
            return tp / (tp + fp) if (tp + fp) != 0 else 0
    
    def recall_score(self, method='macro'):
        tp_per_label = np.sum((self.y_true == 1) & (self.y_pred == 1), axis=0)
        true_sum_per_label = np.sum(self.y_true, axis=0)
        recall_per_label = np.divide(tp_per_label, true_sum_per_label, 
                                    out=np.zeros_like(tp_per_label, dtype=float), 
                                    where=true_sum_per_label != 0)
        
        if method == 'macro':
            return np.mean(recall_per_label)
        elif method == 'micro':
            tp = np.sum((self.y_true == 0) & (self.y_pred == 1))  
            fn = np.sum(self.y_true) - tp
            return tp / (tp + fn) if (tp + fn) != 0 else 0
    
    def f1_score(self, method='macro'):
        precision = self.precision_score(method=method)
        recall = self.recall_score(method=method)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)

    def hamming_loss(self):
        mismatches = np.not_equal(self.y_true, self.y_pred).sum()
        total_labels = self.y_true.shape[0] * self.y_true.shape[1]
        return mismatches / total_labels

