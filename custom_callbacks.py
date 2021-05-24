from keras import backend as K
from keras.callbacks import Callback
from loss_functions import euc_loss1x, euc_loss2x, euc_loss3x, \
    euc_loss1q, euc_loss2q, euc_loss3q, frob_loss1r, frob_loss2r, frob_loss3r
import settings


class LossHistory(Callback):
    def __init__(self, BETA, BETA_list, prev_k, chkp):
        # Save params in constructor
        super().__init__()
        self.BETA = BETA
        self.BETA_list = BETA_list
        self.prev_k = prev_k
        self.chkp = chkp
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        epochs = settings.epochs
        thr = settings.thr
        # loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        x_loss = logs.get('cls3_fc_pose_xyz_loss')
        q_loss = logs.get('cls3_fc_pose_wpqr_loss')
        curr_k = x_loss / q_loss
        best = self.chkp.__dict__.get('best')
        # val_x_loss = logs.get('val_cls3_fc_pose_xyz_loss')
        # val_q_loss = logs.get('val_cls3_fc_pose_wpqr_loss')

        optimizer = self.model.optimizer
        margin = abs(curr_k - self.prev_k)
        if epoch >= int(settings.epochs_proportion * epochs) and (val_loss > best) and (margin <= thr):
            # self.BETA = curr_k * self.BETA
            settings.BETA = self.BETA = curr_k * self.BETA

            # L1 / L2 loss functions (comment/uncomment accordingly)
            # Loss function L1
            self.model.compile(optimizer=optimizer, loss={'cls1_fc_pose_xyz': euc_loss1x,
                                                          'cls1_fc_pose_wpqr': euc_loss1q,
                                                          'cls2_fc_pose_xyz': euc_loss2x,
                                                          'cls2_fc_pose_wpqr': euc_loss2q,
                                                          'cls3_fc_pose_xyz': euc_loss3x,
                                                          'cls3_fc_pose_wpqr': euc_loss3q})

            '''
            # Loss function L2
            self.model.compile(optimizer=optimizer, loss={'cls1_fc_pose_xyz': euc_loss1x,
                                                          'cls1_fc_pose_wpqr': frob_loss1r,
                                                          'cls2_fc_pose_xyz': euc_loss2x,
                                                          'cls2_fc_pose_wpqr': frob_loss2r,
                                                          'cls3_fc_pose_xyz': euc_loss3x,
                                                          'cls3_fc_pose_wpqr': frob_loss3r})
            '''
            # self.model.load_weights(settings.checkpoint_weights)

        n_lr = K.eval(self.model.optimizer.lr)
        self.prev_k = curr_k
        self.BETA_list.append(settings.BETA)
        print('thr = ', thr)
        print('margin = ', margin)
        print('BETA = ', settings.BETA)
        print('lr = ', n_lr)
