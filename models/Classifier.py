import tensorflow as tf

from models import NNBase
from modules.tf_ops import get_hard_labels_from_logits, is_tp_or_tn_op, get_acc


class Classifier(NNBase):
    name = 'classification'
    scopes = ['classifier']
    scope_suffixes = ['_c']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred, self.acc = self.get_pred_and_acc_op(self.y, self.logits)

    @staticmethod
    def get_pred_and_acc_op(labels, logits):
        if len(logits.shape) > 1:
            pred = get_hard_labels_from_logits(logits)
            is_tp_or_tn = is_tp_or_tn_op(labels, pred)
        else:
            pred = tf.cast(logits > 0, tf.float32)
            is_tp_or_tn = pred * labels[:, 0]
        acc = get_acc(labels, is_tp_or_tn)
        return pred, acc

    def get_metric(self, name, **kwargs):
        if name == 'acc':
            return self.acc
        elif name == 'pred':
            return self.pred
        else:
            return super().get_metric(name=name, **kwargs)
