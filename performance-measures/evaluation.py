import numpy as np

class model_evaluation:
    confusion_matrix=None
    def __init__(self, y_true:np.ndarray, y_pred:np.ndarray, classes_list:np.ndarray):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes_list = classes_list

    def accuracy_score(self) -> float:
        correct = (self.y_pred == self.y_true)
        count = np.sum(correct)
        return (count / len(self.y_true))
    
    def calculate_confusion_matrix(self):
        label_to_index = {label: index for index, label in enumerate(self.classes_list)}
        
        classes_cnt = len(self.classes_list)
        self.confusion_matrix = np.zeros((classes_cnt, classes_cnt), dtype=int)

        for yt, yp in zip(self.y_true, self.y_pred):
            yt_idx = label_to_index[yt]
            yp_idx = label_to_index[yp]
            self.confusion_matrix[yp_idx, yt_idx] += 1
        
        return self.confusion_matrix

    def precision_score(self, method='macro') -> float:
        if method != 'macro' and method != 'micro':
            print("Invalid precision method specified")
            exit(1)
        if self.confusion_matrix is None:
            self.calculate_confusion_matrix()
        if method == 'macro':
            classes_cnt = len(self.classes_list)
            p_macro_sum = 0
            for i in range(classes_cnt):
                true_pos = self.confusion_matrix[i, i]
                pred_pos = np.sum(self.confusion_matrix[:, i])

                if pred_pos > 0:
                    p_macro_sum += (true_pos / pred_pos)

            return (p_macro_sum / len(self.y_true))
        
        else:
            total_true_pos = np.diag(self.confusion_matrix).sum()
            total_false_pos = np.sum(self.confusion_matrix) - total_true_pos

            return (total_true_pos / (total_true_pos + total_false_pos))



    def recall_score(self, method='macro') -> float:
        if method != 'macro' and method != 'micro':
            print("Invalid recall method specified")
            exit(1)
        if self.confusion_matrix is None:
            self.calculate_confusion_matrix()

        if method == 'macro':
            classes_cnt = len(self.classes_list)
            r_macro_sum = 0
            for i in range(classes_cnt):
                true_pos = self.confusion_matrix[i, i]
                act_pos = np.sum(self.confusion_matrix[i, :])
                if act_pos > 0:
                    r_macro_sum += (true_pos / act_pos)

            return (r_macro_sum / len(self.y_true))
        else:
            total_true_pos = np.diag(self.confusion_matrix).sum()
            total_false_neg = np.sum(self.confusion_matrix) - total_true_pos

            return (total_true_pos / (total_true_pos + total_false_neg))
        

    def f1_score(self, method='macro') -> float:
        if method != 'macro' and method != 'micro':
            print("Invalid f1-score method specified")
            exit(1)
        if self.confusion_matrix is None:
            self.calculate_confusion_matrix()

        precision = self.precision_score(method)
        recall = self.precision_score(method)
        if precision != 0 and recall != 0:
            return (2 * precision * recall) / (precision + recall)
        else:
            return 0

