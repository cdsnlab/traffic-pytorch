from tensorboardX import SummaryWriter

class Logger():
    def __init__(self) -> None:
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