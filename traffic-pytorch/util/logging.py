import termcolor
import sys

def toRed(content):
    return termcolor.colored(content,"red",attrs=["bold"])

def toGreen(content):
    return termcolor.colored(content,"green",attrs=["bold"])

def toBlue(content):
    return termcolor.colored(content,"blue",attrs=["bold"])

def toCyan(content):
    return termcolor.colored(content,"cyan",attrs=["bold"])

def toYellow(content):
    return termcolor.colored(content,"yellow",attrs=["bold"])

def toMagenta(content):
    return termcolor.colored(content,"magenta",attrs=["bold"])

def toGrey(content):
    return termcolor.colored(content,"grey",attrs=["bold"])

def toWhite(content):
    return termcolor.colored(content,"white",attrs=["bold"])

def print_setup(loss, metric, optimzier, scheduler):
    sys.stdout.write('\tLoss: {}\n\tMetrics: {}\n\tOptimizer: {}\n\tScheduler: {}\n'.format(
        toWhite(loss), toWhite(metric), toWhite(optimzier), toWhite(scheduler)
        )
    )
    
def print_progress(phase, epoch, max_epoch, iter, max_iter, iter_time, loss_name, loss_val, metric_names, metric_vals):
    CURSOR_UP_ONE = '\x1b[1A' 
    ERASE_LINE = '\x1b[2K'

    sys.stdout.write('{}[{}] {} {} {} [{}] {} {} [{}] '.format(
        CURSOR_UP_ONE + ERASE_LINE,
        toWhite(phase),
        toWhite('{} '.format('EP')) + toCyan('{}/{}'.format(epoch, max_epoch)),
        toWhite('{} '.format('ITER')) + toCyan('{}/{}'.format(iter, max_iter)),
        toWhite('ETA') + toGreen(' {:5.2f}s'.format(iter_time * (max_iter-iter))),
        toWhite('LOSS'),
        toWhite(loss_name.replace('_loss', '')),
        '{:.2f}'.format(loss_val),
        toWhite('METRIC')
        )
    )

    for name, val in zip(metric_names, metric_vals):
        sys.stdout.write('{} {:.2f} '.format(toWhite(name.replace('_np','')), val))
    
    sys.stdout.write('\n')

def print_total(phase, epoch, max_epoch, loss_name, loss_val, metric_names, metric_vals):
    sys.stdout.write('[{}] {} [{}] {} {} [{}] '.format(
        toWhite(phase + " TOTAL"),
        toWhite('{} '.format('EP')) + toCyan('{}/{}'.format(epoch, max_epoch)),
        toWhite('AVG LOSS'),
        toWhite(loss_name.replace('_loss', '')),
        '{:.2f}'.format(loss_val),
        toWhite('AVG METRIC')
        )
    )

    for name, val in zip(metric_names, metric_vals):
        sys.stdout.write('{} {:.2f} '.format(toWhite(name.replace('_np','')), val))
    
    sys.stdout.write('\n\n')