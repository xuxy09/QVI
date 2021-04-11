exp_name = 'qvi'
record_dir = 'records/{}'.format(exp_name)
checkpoint_dir = 'checkpoints/{}'.format(exp_name)
trainset = 'QVI960'
trainset_root = './datasets/QVI-960'
train_size = (640, 360)
train_crop_size = (355, 355)

validationset = 'Adobe240all'
validationset_root = './datasets/Adobe240_validation'
validation_size = (640, 360)
validation_crop_size = (640, 360)

train_batch_size = 25

train_continue = False
epochs = 250
progress_iter = 200
checkpoint_epoch = 5


mean = [0.429, 0.431, 0.397]
std  = [1, 1, 1]

model = 'QVI'
pwc_path = './utils/pwc-checkpoint.pt'

init_learning_rate = 1e-4
milestones = [100, 150]

