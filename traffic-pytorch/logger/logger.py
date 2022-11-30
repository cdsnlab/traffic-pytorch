from tensorboardX import SummaryWriter

class Logger():
    def __init__(self, save_name = None) -> None:
        if save_name is not None:
            self.logger = SummaryWriter(logdir='runs/{}'.format(save_name))
        else:
            self.logger = SummaryWriter()

    # TODO: Add more methods to log training and validation metrics
    def log_training(self, loss, metrics, epoch):
        self.logger.add_scalars('loss',{
                'trainiing_loss': loss
            }, epoch)
        self.logger.add_scalars('metrics',{
                'trainiing_metrics': metrics
        }, epoch)

    def log_validation(self, loss, metrics, epoch):
        self.logger.add_scalars('loss',{
                'validation_loss': loss,
            }, epoch)
        self.logger.add_scalars('metrics',{
                'validation_metrics': metrics
        }, epoch)
    
    def close(self):
        self.logger.close()